OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG)
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm lorenz_ode.c \
    	  -o lorenz_ode.ll
	$(OPT) -passes=error-analysis -S lorenz_ode.ll \
								  -o lorenz_ode_ac.ll

run_pass:
	$(LLI) lorenz_ode_ac.ll

build_original:
	gcc lorenz_ode.c -lm

clean:
	rm -rf lorenz_ode.ll lorenz_ode_ac.ll .fAC_logs a.out *.txt
