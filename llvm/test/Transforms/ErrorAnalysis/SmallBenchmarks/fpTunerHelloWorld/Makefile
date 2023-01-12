OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(NO_DATA_DUMP) $(MEMORY_OPT) -g
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [NO_DATA_DUMP=-DNO_DATA_DUMP] [MEMORYOPT=-DMEMORY_OPT]
build_pass_2:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm fptunerhelloworld0.c \
    	  -o fptunerhelloworld0.ll
	$(OPT) -passes=error-analysis -S fptunerhelloworld0.ll \
								  -o fptunerhelloworld0_ac.ll

run_pass_2:
	$(LLI) fptunerhelloworld0_ac.ll


build_original:
	g++ fptunerhelloworld0.cpp -lm -o fptunerhelloworld0.out

clean:
	rm -rf *.ll .fAF_logs fptunerhelloworld0.out *.txt
