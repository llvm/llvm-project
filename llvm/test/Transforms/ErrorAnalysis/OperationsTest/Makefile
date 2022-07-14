OP = 	-O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG)
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm operations.c \
    	  -o operations.ll
	$(OPT) -passes=error-analysis -S operations.ll \
								  -o operations_ac.ll

run_pass:
	$(LLI) operations_ac.ll

build_original:
	gcc operations.c -lm

clean:
	rm -rf operations.ll operations_ac.ll .fAC_logs a.out
