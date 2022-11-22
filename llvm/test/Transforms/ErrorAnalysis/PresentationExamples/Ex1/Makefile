OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(DUMP_SYMBOLIC) $(MEMORY_OPT)
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [DUMP_SYMBOLIC=-DFAF_DUMP_SYMBOLIC] [MEMORYOPT=-DMEMORY_OPT]
build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm ex1.c \
    	  -o ex1.ll
	$(OPT) -passes=error-analysis -S ex1.ll \
								  -o ex1_ac.ll

run_pass:
	$(LLI) ex1_ac.ll

build_original:
	gcc ex1.c -lm -o ex1.out

clean:
	rm -rf .fAF_logs a.out ex1.ll ex1_ac.ll ex1.out
