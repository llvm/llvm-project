OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(NO_DATA_DUMP) $(MEMORY_OPT) -g
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [NO_DATA_DUMP=-DNO_DATA_DUMP] [MEMORYOPT=-DMEMORY_OPT]
build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm cg_inline.cpp \
    	  -o cg_inline.ll
	$(OPT) -passes=error-analysis -S cg_inline.ll \
								  -o cg_inline_ac.ll


run_pass:
	$(LLI) cg_inline_ac.ll


build_original:
	g++ cg_inline.cpp -lm -o cg_inline.out

clean:
	rm -rf cg_inline.ll cg_inline_ac.ll .fAF_logs a.out *.txt
