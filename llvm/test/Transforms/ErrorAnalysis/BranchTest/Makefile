OP = 	-O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(DUMP_SYMBOLIC)
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [DUMP_SYMBOLIC=-DFAF_DUMP_SYMBOLIC]
build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/ComputationGraph.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/RandomTesting/URT.h \
	$(OP) -S -emit-llvm IfConditional.c \
    	  -o IfConditional.ll
	$(OPT) -passes=error-analysis -S IfConditional.ll \
								  -o IfConditional_ac.ll

run_pass:
	$(LLI) IfConditional_ac.ll

build_original:
	gcc IfConditional.c

clean:
	rm -rf IfConditional.ll IfConditional_ac.ll .fAC_logs a.out
