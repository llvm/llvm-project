OP = 	-O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h \
	$(OP) -S -emit-llvm forLoop.c \
    	  -o forLoop.ll
	$(OPT) -passes=error-analysis -S forLoop.ll \
								  -o forLoop_ac.ll

run_pass:
	$(LLI) forLoop_ac.ll

build_original:
	gcc forLoop.c

clean:
	rm -rf forLoop.ll forLoop_ac.ll .fAC_logs a.out
