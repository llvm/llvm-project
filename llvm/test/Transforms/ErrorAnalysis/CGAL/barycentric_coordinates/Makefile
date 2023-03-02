OP =  -O0
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(NO_DATA_DUMP) $(MEMORY_OPT) -g
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [NO_DATA_DUMP=-DNO_DATA_DUMP] [MEMORYOPT=-DMEMORY_OPT]
build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm barycentric_coordinates.c \
    	  -o barycentric_coordinates.ll
	$(OPT) -passes=error-analysis -S barycentric_coordinates.ll \
								  -o barycentric_coordinates_ac.ll

run_pass:
	$(LLI) barycentric_coordinates_ac.ll


build_original:
	clang barycentric_coordinates.c $(OP) -lm -o barycentric_coordinates.out

clean:
	rm -rf *.ll .fAF_logs barycentric_coordinates.out *.txt
