using Multilane
using POMDPs

solver = SimpleSolver()

cor = 0.75 # correlation of model parameters

behaviors = standard_uniform(correlation=cor)
pp = PhysicalParam(4, lane_length=100.0)

dmodel = NoCrashIDMMOBILModel(10, pp,behaviors=behaviors,p_appear=1.0,
			      lane_terminate=true,max_dist=1000.0,
			      brake_terminate_thresh=4.0,speed_terminate_thresh=15.0)

rmodel = SuccessReward(lambda=0)

pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)

mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)

policy = solve(solver, mdp)

## This code won't actually be run; it's just to show you how to make a MLPhysicalState. 
## The actual values will be from the python code
seed = parse(Int64,ARGS[1])
println("seed = $seed")


srand(seed)
n = 5
# x locations of n cars
xs = rand(5).* 100.0
# y locations of n cars # in lane coordinates; not 
ys = rand(5).* 4
# velocities of n cars
vels = 1.0*randn(5).+ 30.0

x = 150.0 # distance of ego car down the road
t = x/30.0 # elapsed time
egostate = CarPhysicalState(50.0, 0.0, 30.0, 0.0, 1) # ego state always has the same x because  
state = MLPhysicalState(x, t, [egostate], nothing)
for i in 1:n
	cs = CarPhysicalState(xs[i], ys[i], vels[i], 0.0, i+1)
	        push!(state.cars, cs)
	end

# get an action from the policy
a = action(policy, state)

@show a

# From Zach meeting for video making and running an entire sim using the policy
# instead of just finding an action as is being done now
#for (s, a) in stepthough(mdp, policy, "s,a")
#	visualize(mdp, s)
	# Will need to  save as png
#end

# Useful files from Zach meeting
#scratch18_04/make_video.jl
#src/visualize.jl
