OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(MEMORY_OPT) -g
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/lib/Transforms/ErrorAnalysis/Utilities/RunTimeUtilities.h \
	$(OP) -S -emit-llvm forLoop.c \
    	  -o forLoop.ll
	$(OPT) -passes=error-analysis -S forLoop.ll \
								  -o forLoop_ac.ll \
								  -function-to-analyze temp \
								  -evaluations 10

run_pass:
	$(LLI) forLoop_ac.ll

build_original:
	gcc forLoop.c -o forLoop.out

clean:
	rm -rf .fAC_logs a.out forLoop.ll forLoop_ac.ll forLoop.out
