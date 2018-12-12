# ----Not sure whether to have the bottom in AutoEnvs.jl or here--------
#using POMDPs
#using Multilane
#--------------------------------------------------------------------

export 
    MultiagentNGSIMEnvVideoMaker,
    reset,
    step,
    observation_space_spec,
    action_space_spec,
    obs_names,
    render

#=
Description:
    Multiagent NGSIM env that plays NGSIM trajectories, allowing a variable 
    number of agents to simultaneously control vehicles in the scene

    Raunak: This is basically a copy of multiagent_ngsim_env.jl with just 
    a few additions to enable color coded video making
=#
type MultiagentNGSIMEnvVideoMaker <: Env
    trajdatas::Vector{ListRecord}
    trajinfos::Vector{Dict}
    roadways::Vector{Roadway}
    roadway::Union{Void, Roadway} # current roadway
    scene::Scene
    rec::SceneRecord
    ext::MultiFeatureExtractor
    egoids::Vector{Int} # current ids of relevant ego vehicles
    ego_vehs::Vector{Union{Void, Vehicle}} # the ego vehicles
    traj_idx::Int # current index into trajdatas 
    t::Int # current timestep in the trajdata
    h::Int # current maximum horizon for egoid
    H::Int # maximum horizon
    primesteps::Int # timesteps to prime the scene
    Δt::Float64

    # multiagent type members
    n_veh::Int # number of simultaneous agents
    remove_ngsim_veh::Bool # whether to remove ngsim veh from all scenes
    features::Array{Float64}

    # metadata
    epid::Int # episode id
    render_params::Dict # rendering options
    infos_cache::Dict # cache for infos intermediate results

    #----- Zach stuff----
    solver::Solver
    cor::Float64
    behaviors
    pp::PhysicalParam
    dmodel::NoCrashIDMMOBILModel
    rmodel::SuccessReward
    pomdp::NoCrashPOMDP
    mdp::NoCrashMDP
    policy
    ##--------------------


    
    function MultiagentNGSIMEnvVideoMaker(
            params::Dict; 
            trajdatas::Union{Void, Vector{ListRecord}} = nothing,
            trajinfos::Union{Void, Vector{Dict}} = nothing,
            roadways::Union{Void, Vector{Roadway}} = nothing,
            reclength::Int = 5,
            Δt::Float64 = .1,
            primesteps::Int = 50,
            H::Int = 50,
            n_veh::Int = 20,
            remove_ngsim_veh::Bool = false,
            render_params::Dict = Dict("zoom"=>5., "viz_dir"=>"/tmp") 
	    )
        
	    
	param_keys = keys(params)
        @assert in("trajectory_filepaths", param_keys)

        # optionally overwrite defaults
        reclength = get(params, "reclength", reclength)
        primesteps = get(params, "primesteps", primesteps)
        H = get(params, "H", H)
        n_veh = get(params, "n_veh", n_veh)
        remove_ngsim_veh = get(params, "remove_ngsim_veh", remove_ngsim_veh)
        for (k,v) in get(params, "render_params", render_params)
            render_params[k] = v
        end

        # load trajdatas if not provided
        if trajdatas == nothing || trajinfos == nothing || roadways == nothing
            trajdatas, trajinfos, roadways = load_ngsim_trajdatas(
                params["trajectory_filepaths"],
                minlength=primesteps + H
            )
        end

        # build components
        scene_length = max_n_objects(trajdatas)
        scene = Scene(scene_length)
        rec = SceneRecord(reclength, Δt, scene_length)
        ext = build_feature_extractor(params)
        infos_cache = fill_infos_cache(ext)
        # features are stored in row-major order because they will be transferred
        # to python; this is inefficient in julia, but don't change or it will 
        # break the python side of the interaction
        features = zeros(n_veh, length(ext))
        egoids = zeros(n_veh)
        ego_vehs = [nothing for _ in 1:n_veh]

	#--------------Zach stuff
	solver_choice = "heuristic"

		#------------- Heuristic
	
	if solver_choice == "heuristic"
		@show "Heuristic solver being used \n"
		solver = SimpleSolver()
		# """SimpleSolver is defined in Multilane.jl/src/heuristics.jl"""
	else
		@show "MCTS with DPW being used \n"
		#------------- MCTS with DPW assuming everyone else is normal
		dpws = DPWSolver(depth = 40,n_iterations = 1000,max_time=Inf,
				 exploration_constant=8.0,k_state=4.5,
				 alpha_state=1/10.0,enable_action_pw=false,
				 check_repeat_state=false,
				 estimate_value=RolloutEstimator(
						SimpleSolver())
				 )
		solver = SingleBehaviorSolver(dpws,Multilane.NORMAL)
		#-----------------------------------------
	end

	cor = 0.75
	behaviors = standard_uniform(correlation=cor)
	pp = PhysicalParam(4, lane_length=100.0)
	dmodel = NoCrashIDMMOBILModel(10,pp,behaviors=behaviors,p_appear=1.0,
				      lane_terminate=true,max_dist=1000.0,
				      brake_terminate_thresh=4.0,
				      speed_terminate_thresh=15.0)
	rmodel = SuccessReward(lambda=0)
	pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)
	mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)
	policy = solve(solver,mdp)

	#--------------------------------


        return new(
            trajdatas, 
            trajinfos, 
            roadways,
            nothing,
            scene, 
            rec, 
            ext, 
            egoids, ego_vehs, 0, 0, 0, H, primesteps, Δt, 
            n_veh, remove_ngsim_veh, features,
            0, render_params, infos_cache
	    #-----------Zach-------------------------------------------
	    ,solver,cor,behaviors,pp,dmodel,rmodel,pomdp,mdp,policy
	    #----------------------------------------------------------
        )

	#----------------Zach-------------------------------------------
	# Raunak: Giving initial value to these guys here has no meaning
	# Give them either in the constructor argument
	# Or include them in the return thing from this function


	#solver = SimpleSolver()
	#cor = 0.75
	#behaviors = standard_uniform(correlation=cor)
	#pp = PhysicalParam(4, lane_length=100.0)
	#dmodel = NoCrashIDMMOBILModel(10,pp,behaviors=behaviors,p_appear=1.0,
	#			      lane_terminate=true,max_dist=1000.0,
	#			      brake_terminate_thresh=4.0,
	#			      speed_terminate_thresh=15.0)
	#rmodel = SuccessReward(lambda=0)
	#pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)
	#mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)
	#policy = sovle(solver,mdp)
	##-----------------------------------------------------------------
    end
end

#=
Description:
    Reset the environment. Note that this environment maintains the following 
    invariant attribute: at any given point, all vehicles currently being controlled
    will end their episode at the same time. This simplifies the rest of the code 
    by enforcing synchronized restarts, but it does somewhat limit the sets of 
    possible vehicles that can simultaneously interact. With a small enough minimum 
    horizon (H <= 250 = 25 seconds) and number of vehicle (n_veh <= 100)
    this should not be a problem. If you need to run with larger numbers then those 
    implement an environment with asynchronous resets.

Args:
    - env: env to reset 
    - dones: bool vector indicating which indices have reached a terminal state 
        these must be either all true or all false
=#
function reset(
        env::MultiagentNGSIMEnvVideoMaker,
        dones::Vector{Bool} = fill!(Vector{Bool}(env.n_veh), true); 
        offset::Int=env.H + env.primesteps,
        random_seed::Union{Void, Int} = nothing)
    # enforce environment invariant reset property 
    # (i.e., always either all true or all false)
    @assert (all(dones) || all(.!dones))
    # first == all at this point, so if first is false, skip the reset
    if !dones[1]
        return
    end

    # sample multiple ego vehicles
    # as stated above, these will all end at the same timestep
    env.traj_idx, env.egoids, env.t, env.h = sample_multiple_trajdata_vehicle(
        env.n_veh,
        env.trajinfos, 
        offset,
        rseed=random_seed
    )  

    # update / reset containers
    env.epid += 1
    empty!(env.rec)
    empty!(env.scene)
    
    # prime 
    for t in env.t:(env.t + env.primesteps)
        get!(env.scene, env.trajdatas[env.traj_idx], t)
        if env.remove_ngsim_veh
            keep_vehicle_subset!(env.scene, env.egoids)
        end
        update!(env.rec, env.scene)
    end

    # set the ego vehicle
    for (i, egoid) in enumerate(env.egoids)
        vehidx = findfirst(env.scene, egoid)
        env.ego_vehs[i] = env.scene[vehidx]
    end
    # set the roadway
    env.roadway = env.roadways[env.traj_idx]
    # env.t is the next timestep to load
    env.t += env.primesteps + 1
    # enforce a maximum horizon 
    env.h = min(env.h, env.t + env.H)
    return get_features(env)
end 

#=
Description:
    Propagate a single vehicle through an otherwise predeterined trajdata

Args:
    - env: environment to be stepped forward
    - action: array of floats that can be converted into an AccelTurnrate
=#
function _step!(env::MultiagentNGSIMEnvVideoMaker, action::Array{Float64})
    # make sure number of actions passed in equals number of vehicles
    @assert size(action, 1) == env.n_veh
    ego_states = Vector{VehicleState}(env.n_veh)
    
    #@show size(action)
    #println("Here is the action from _step!\n")
    #@show action

    # propagate all the vehicles and get their new states
    for (i, ego_veh) in enumerate(env.ego_vehs)
        # convert action into form 
	ego_action = AccelTurnrate(action[i,:]...)
	
	
	#--------------------ZACH--------------------------
	
	if i==1
		#println(ego_veh.state.posG.x,",",ego_veh.state.posG.y,",",ego_veh.state.posG.θ)
		# """ To enable plotting on a spreadsheet"""
		
		#println(ego_veh.state.posF.s)
		# """s gives distance along lane in the Frenet frame"""
		
		
		# """Sample test code from Zach
		#seed = 15
		#srand(seed)
		#n=5
		#xs=rand(5).*100.0
		#ys=rand(5).*4
		#vels=1.0*randn(5).+30.0

		#x=150.0
		#t = x/30.0
		
		
		# TODO Get the x position of the ego vehicle (properly)
		ego_x = ego_veh.state.posF.s
		planner_t = env.t

		# Will be needed for relative frenet
		ego_roadind = ego_veh.state.posF.roadind		
		
		# Will be needed to check if in same segment
		ego_segment = ego_veh.state.posF.roadind.tag.segment

		# Get ego y position according to Zach frame
		ego_lanenum = ego_veh.state.posF.roadind.tag.lane
		ego_offset = ego_veh.state.posF.t
		lanewidth = get_lane_width(ego_veh,env.roadway)
		y_egoveh = (ego_lanenum-1)*lanewidth+ego_offset



		planner_egostate = CarPhysicalState(
					50.0,y_egoveh,ego_veh.state.v,0.0,1)
		# """CarPhysicalState is defined in Multilane.jl/src/MDP_types.jl
		# The 5 elements in the struct are
		# x (longitude), y (lane number), velocity, lane_change, id"""


		planner_state=MLPhysicalState(
					ego_x,planner_t,[planner_egostate],nothing)
		# """MLPhysicalState is defined in Multilane.jl/src/MDP_types.jl
		# The 4 elements in the struct are
		# x (longitude), t (time), cars (an array of CarPhysicalStates,terminal"""

		#for j in 1:n
		#	cs = CarPhysicalState(xs[j], ys[j], vels[j], 0.0, j+1)
		#	push!(planner_state.cars,cs)
		#end
	
		
		countVeh = 0
		totalVeh = 0
		planner_id_count = 1	# Functions like j in Zach's example code
		
		# Loop over all the vehicles in the scene and insert those
		# information who are within 50m radius of ego vehicle
		for veh in env.scene
			totalVeh+=1 #counts total number of vehs in scene
			#@show totalVeh
			
			# TODO Check if hypot is still used in the updated
			# form of Vec.jl
			distance = hypot(ego_veh.state.posG - veh.state.posG)
			
			# Find cars within a radius of 50 m from Zach
			# Also check that ego_veh does not consider itself in the planner
			if distance < 50 && veh != ego_veh
				countVeh+=1

				# Need to find the longitudinal distance of
				# veh relative to 50 m behind ego and find the 
				# lane unit relative to same point
				# Refer ZachStateSpace.jpg in DailyNotes/September
				
				# """From projections.ipynb"""
				# Check whether veh in same road segment as ego
				veh_segment = veh.state.posF.roadind.tag.segment
				veh_global = veh.state.posG
				if veh_segment == ego_segment
					#@show "yomahesh"
					
					#"""get_frenet_relative_position
					# is defined in src/2D/vehicles/scenes.jl
					#Fields in relfre are
					# origin, target, Delta_s, t, phi"""
					relfre = get_frenet_relative_position(
							veh_global,ego_roadind,
							env.roadway)
					
					
					#"""Positive if veh is front of ego"""
					xdist = relfre.Δs
					
					#"""Positive if veh is left of ego"""
					ydist = relfre.t
					
					#""" Transform to the reference frame of the
					# MLPhysicalState
					# Origin is 50 m behind ego and end of rightmost
					# lane on the road"""
					planner_x = xdist+50
					
					planner_y = y_egoveh+ydist
					#@show planner_x, planner_y
					
					cs = CarPhysicalState(
						planner_x, planner_y, veh.state.v, 0.0,
						planner_id_count+1)
					
					planner_id_count += 1
					push!(planner_state.cars,cs)

				end #Finish check to see if in same segment

			end # Finish check to see if within 50m radius
		end # Finish looping over all the vehicles in the scene
		#@show countVeh
		#@show totalVeh
		@show planner_state
		planner_action = POMDPs.action(env.policy,planner_state)
		# planner_action is of type multilane.MLAction
		# It has acc and ydot as the two things in it
		
		planner_accl = planner_action.acc
		planner_latvel = planner_action.lane_change
		#@show typeof(planner_action)
		@show planner_action
		#@show planner_accl
		#@show planner_ydot

		v = ego_veh.state.v
		phi = ego_veh.state.posF.ϕ
		dt = 1 #TODO Investigate why env.Δt does not do well here (causes oscillation)
		lat_accel = (2/(dt*dt))*(planner_latvel*dt - v*sin(phi)*dt)

		ego_action = LatLonAccel([lat_accel, planner_accl]...)
		
	end #end check to see if this vehicle is driven using Zach
	#--------------------------------------------------

	# propagate the ego vehicle 
        ego_states[i] = propagate(
            ego_veh, 
            ego_action, 
            env.roadway, 
            env.Δt
        )
	
        # update the ego_veh
        env.ego_vehs[i] = Entity(ego_veh, ego_states[i])
    end

    #@show env.scene
    #@show env.t


    # load the actual scene, and insert the vehicles into it
    get!(env.scene, env.trajdatas[env.traj_idx], env.t)
    if env.remove_ngsim_veh
        keep_vehicle_subset!(env.scene, env.egoids)
    end
    orig_vehs = Vector{Vehicle}(env.n_veh)

    for (i, egoid) in enumerate(env.egoids)

	vehidx = findfirst(env.scene, egoid)

        # track the original vehicle for validation / infos purposes
        orig_vehs[i] = env.scene[vehidx]

	# replace the original with the controlled vehicle

        env.scene[vehidx] = env.ego_vehs[i]

    end

    # update rec with current scene 
    update!(env.rec, env.scene)

    # Raunak adds in original vehicle properties
    step_infos = Dict{String, Vector{Float64}}(
        "rmse_pos"=>Float64[],
        "rmse_vel"=>Float64[],
        "rmse_t"=>Float64[],
        "x"=>Float64[],
        "y"=>Float64[],
        "s"=>Float64[],
        "phi"=>Float64[],
	"orig_x"=>Float64[],
	"orig_y"=> Float64[],
	"orig_theta"=>Float64[],
	"orig_length"=>Float64[],
	"orig_width"=>Float64[]
    )
    for i in 1:env.n_veh
        push!(step_infos["rmse_pos"], sqrt(abs2((orig_vehs[i].state.posG - env.ego_vehs[i].state.posG))))
        push!(step_infos["rmse_vel"], sqrt(abs2((orig_vehs[i].state.v - env.ego_vehs[i].state.v))))
        push!(step_infos["rmse_t"], sqrt(abs2((orig_vehs[i].state.posF.t - env.ego_vehs[i].state.posF.t))))
        push!(step_infos["x"], env.ego_vehs[i].state.posG.x)
        push!(step_infos["y"], env.ego_vehs[i].state.posG.y)
        push!(step_infos["s"], env.ego_vehs[i].state.posF.s)
        push!(step_infos["phi"], env.ego_vehs[i].state.posF.ϕ)
	push!(step_infos["orig_x"], orig_vehs[i].state.posG.x)
	push!(step_infos["orig_y"], orig_vehs[i].state.posG.y)
	push!(step_infos["orig_theta"], orig_vehs[i].state.posG.θ)
	push!(step_infos["orig_length"], orig_vehs[i].def.length)
	push!(step_infos["orig_width"], orig_vehs[i].def.width)
    end

    return step_infos
end

function _extract_rewards(env::MultiagentNGSIMEnvVideoMaker, infos::Dict{String, Array{Float64}})
    rewards = zeros(env.n_veh)
    R = 0
    
    for i in 1:env.n_veh
        if infos["is_colliding"][i] == 1
            rewards[i] -= R
        elseif infos["is_offroad"][i] == 1
            rewards[i] -= R
        elseif infos["hard_brake"][i] == 1
            rewards[i] -= (R*0.5) # braking hard is not as bad as a collision
        end
    end
    return rewards
end

function Base.step(env::MultiagentNGSIMEnvVideoMaker, action::Array{Float64})
	#println("\nHere is the action from step")
	#@show action
	
	step_infos = _step!(env, action)
    
	# compute features and feature_infos 
	features = get_features(env)
	feature_infos = _compute_feature_infos(env, features)
    
	# combine infos 
	infos = merge(step_infos, feature_infos)
    
	# update env timestep to be the next scene to load
	env.t += 1
    
    # compute terminal
    terminal = env.t > env.h ? true : false
    terminal = [terminal for _ in 1:env.n_veh]
    # vectorized sampler does not call reset on the environment
    # but expects the environment to handle resetting, so do that here
    # note: this mutates env.features in order to return the correct obs when resetting
    reset(env, terminal)
	rewards = _extract_rewards(env, infos)
    return deepcopy(env.features), rewards, terminal, infos
end

function _compute_feature_infos(env::MultiagentNGSIMEnvVideoMaker, features::Array{Float64};
                                                         accel_thresh::Float64=-3.0)
    feature_infos = Dict{String, Array{Float64}}(
                "is_colliding"=>Float64[], 
                "is_offroad"=>Float64[],
                "hard_brake"=>Float64[],
		"colliding_veh_ids"=>Float64[],
		"offroad_veh_ids"=>Float64[],
		"hardbrake_veh_ids"=>Float64[]
		)

    # Raunak explains: env.n_veh will be number of policy driven cars
    # Caution: This need not be the same as number of cars in the scene
    # Because the scene contains both policy driven cars and ngsim replay cars

    # Raunak debugging
    #sizeFeatures=size(features)
    #println("Compute repoting featurs are \n $sizeFeatures")
    #something = env.infos_cache
    #println("env.infos_cache =\n $something")
    #somethingelse = env.infos_cache["is_colliding_idx"]
    #println("env.infos_cache[is_colliding_idx]=$somethingelse")

    for i in 1:env.n_veh
	is_colliding = features[i, env.infos_cache["is_colliding_idx"]]
	#println("is_colliding=$is_colliding\n")
	is_offroad = features[i, env.infos_cache["out_of_lane_idx"]]
        accel = features[i, env.infos_cache["accel_idx"]]
        push!(feature_infos["hard_brake"], accel <= accel_thresh)
        push!(feature_infos["is_colliding"], is_colliding)
        push!(feature_infos["is_offroad"], is_offroad)
	
	# Raunak adding list of colliding ego ids into the feature list that gets passed to render
	if is_colliding==1
		push!(feature_infos["colliding_veh_ids"],env.ego_vehs[i].id)
		#println("Collision has happened see red")
	end
	if is_offroad==1
		push!(feature_infos["offroad_veh_ids"],env.ego_vehs[i].id)
		#println("Offroad has happened see yellow")
	end
	if accel <= accel_thresh
		push!(feature_infos["hardbrake_veh_ids"],env.ego_vehs[i].id)
		#println("Hard brake has happened see some color")
	end
    end
    return feature_infos
end


function AutoRisk.get_features(env::MultiagentNGSIMEnvVideoMaker)
    for (i, egoid) in enumerate(env.egoids)
	veh_idx = findfirst(env.scene, egoid)
	pull_features!(env.ext, env.rec, env.roadway, veh_idx)
        env.features[i, :] = deepcopy(env.ext.features)
    end
    return deepcopy(env.features)
end


function observation_space_spec(env::MultiagentNGSIMEnvVideoMaker)
    low = zeros(length(env.ext))
    high = zeros(length(env.ext))
    feature_infos = feature_info(env.ext)
    for (i, fn) in enumerate(feature_names(env.ext))
        low[i] = feature_infos[fn]["low"]
        high[i] = feature_infos[fn]["high"]
    end
    infos = Dict("high"=>high, "low"=>low)
    return (length(env.ext),), "Box", infos
end
action_space_spec(env::MultiagentNGSIMEnvVideoMaker) = (2,), "Box", Dict("high"=>[4.,.15], "low"=>[-4.,-.15])
obs_names(env::MultiagentNGSIMEnvVideoMaker) = feature_names(env.ext)
vectorized(env::MultiagentNGSIMEnvVideoMaker) = true
num_envs(env::MultiagentNGSIMEnvVideoMaker) = env.n_veh

# Raunak defined this render function to enable color changing based on collisions and offroads
function render(
        env::MultiagentNGSIMEnvVideoMaker;
	infos=Dict(),
        egocolor::Vector{Float64}=[0.,0.,1.],
        camtype::String="follow",
        static_camera_pos::Vector{Float64}=[0.,0.],
        camera_rotation::Float64=0.,
        canvas_height::Int=800,
        canvas_width::Int=800)
    # define colors for all the vehicles
    carcolors = Dict{Int,Colorant}()
    egocolor = ColorTypes.RGB(egocolor...)

    # Loop over all the vehicles in the scene. Note these may be both policy driven and ngsim replay
    for veh in env.scene
	# If current vehicle is a policy driven vehicle then color it blue otherwise color it green
	carcolors[veh.id] = in(veh.id, env.egoids) ? egocolor : colorant"green"
	
	# ----------------ZACH------------------------
	if veh.id == 2582
		carcolors[veh.id] = colorant"cyan"
		#println("Coloring Zach car yellow")
	end
	#--------------------------------------

	# If the current vehicle is in the list of colliding vehicles color it red
	if in(veh.id,infos["colliding_veh_ids"])
		carcolors[veh.id] = ColorTypes.RGB([1.,0.,0.]...)
	end

	# If current vehicle is in the list of offroad vehicles color it yellow
	if in(veh.id,infos["offroad_veh_ids"])
		carcolors[veh.id]=ColorTypes.RGB([1.,1.,0.]...)
	end

	# If current vehicle is in the list of hard brakers then color it light blue
	#if in(veh.id,infos["hardbrake_veh_ids"])
	#	carcolors[veh.id]=ColorTypes.RGB([0.,1.,1.]...)
	#end
    end

    # define a camera following the ego vehicle
    if camtype == "follow"
        # follow the first vehicle in the scene
        cam = AutoViz.CarFollowCamera{Int}(env.egoids[1], env.render_params["zoom"])
    elseif camtype == "static"
        cam = AutoViz.StaticCamera(VecE2(static_camera_pos...), env.render_params["zoom"])
    else
        error("invalid camera type $(camtype)")
    end
    
    # Raunak commented this out because it was creating rays that were being used for
    # some research that Tim had been doing
    overlays = [
        CarFollowingStatsOverlay(env.egoids[1], 2), 
    #    NeighborsOverlay(env.egoids[1], textparams = TextParams(x = 600, y_start=300))
    ]

    # Raunak video plotting the ghost vehicle
    # See 'OrigVehicleOverlay' defined in AutoViz/src/2d/overlays.jl
    # to understand how the ghost vehicle is being plotted
#    overlays = [
#       CarFollowingStatsOverlay(env.egoids[1], 2)
#	,OrigVehicleOverlay(infos["orig_x"][1],infos["orig_y"][1],infos["orig_theta"][1],infos["orig_length"][1],infos["orig_width"][1])
#    ]


    # rendermodel for optional rotation
    # note that for this to work, you have to comment out a line in AutoViz
    # src/overlays.jl:27 `clear_setup!(rendermodel)` in render
    rendermodel = RenderModel()
    camera_rotate!(rendermodel, deg2rad(camera_rotation))

    # render the frame
    frame = render(
        env.scene, 
        env.roadway,
        overlays, 
        rendermodel = rendermodel,
        cam = cam, 
        car_colors = carcolors,
        canvas_height=canvas_height,
        canvas_width=canvas_width
    )

    # save the frame 
    if !isdir(env.render_params["viz_dir"])
        mkdir(env.render_params["viz_dir"])
    end
    ep_dir = joinpath(env.render_params["viz_dir"], "episode_$(env.epid)")
    if !isdir(ep_dir)
        mkdir(ep_dir)
    end
    filepath = joinpath(ep_dir, "step_$(env.t).png")
    write_to_png(frame, filepath)

    # load and return the frame as an rgb array
    img = PyPlot.imread(filepath)
    return img
end
