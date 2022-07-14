OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG)
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm forLoop.c \
    	  -o forLoop.ll
	$(OPT) -passes=error-analysis -S forLoop.ll \
								  -o forLoop_ac.ll
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
  		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h \
  		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
  	$(OP) -S -emit-llvm doubleForLoop.c \
      	  -o doubleForLoop.ll
	$(OPT) -passes=error-analysis -S doubleForLoop.ll \
    	  						  -o doubleForLoop_ac.ll

run_pass1:
	$(LLI) forLoop_ac.ll

run_pass2:
	$(LLI) doubleForLoop_ac.ll

build_original:
	gcc forLoop.c -o forLoop.out
	gcc doubleForLoop.c -o doubleForLoop.out

clean:
	rm -rf .fAC_logs a.out forLoop.ll forLoop_ac.ll forLoop.out doubleForLoop.ll doubleForLoop_ac.ll doubleForLoop.out
