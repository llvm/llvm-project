OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(NO_DATA_DUMP) $(MEMORY_OPT) -g
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [NO_DATA_DUMP=-DNO_DATA_DUMP] [MEMORYOPT=-DMEMORY_OPT]
build_pass_2:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm cg_destructured_2.cpp \
    	  -o cg_destructured_2.ll
	$(OPT) -passes=error-analysis -S cg_destructured_2.ll \
								  -o cg_destructured_2_ac.ll

run_pass_2:
	$(LLI) cg_destructured_2_ac.ll


build_original:
	g++ cg_destructured_2.cpp -lm -o cg_destructured_2.out

clean:
	rm -rf *.ll .fAF_logs cg_destructured_2.out *.txt
