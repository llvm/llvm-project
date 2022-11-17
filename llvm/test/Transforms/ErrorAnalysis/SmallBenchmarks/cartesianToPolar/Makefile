OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(DUMP_SYMBOLIC)
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [DUMP_SYMBOLIC=-DFAF_DUMP_SYMBOLIC]
build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
	--include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm cartesianToPolar.c \
    	  -o cartesianToPolar.ll
	$(OPT) -passes=error-analysis -S cartesianToPolar.ll \
								  -o cartesianToPolar_ac.ll

run_pass:
	$(LLI) cartesianToPolar_ac.ll

build_original:
	gcc cartesianToPolar.c -lm

clean:
	rm -rf cartesianToPolar.ll cartesianToPolar_ac.ll .fAF_logs a.out
