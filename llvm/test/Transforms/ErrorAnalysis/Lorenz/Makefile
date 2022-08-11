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
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
  		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h \
  		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
  	$(OP) -S -emit-llvm lorenz_simple.c \
      	  -o lorenz_simple.ll
	$(OPT) -passes=error-analysis -S lorenz_simple.ll \
    	  						  -o lorenz_simple_ac.ll

run_pass1:
	$(LLI) lorenz_ode_ac.ll

run_pass2:
	$(LLI) lorenz_simple_ac.ll

build_original:
	gcc lorenz_ode.c -lm -o lorenz_ode.out
	gcc lorenz_simple.c -lm -o lorenz_simple.out

clean:
	rm -rf lorenz_ode.ll lorenz_ode_ac.ll lorenz_simple.ll lorenz_simple_ac.ll .fAC_logs a.out *.txt
