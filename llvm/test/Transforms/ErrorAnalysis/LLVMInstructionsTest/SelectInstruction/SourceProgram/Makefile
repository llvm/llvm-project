OP =  -O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang $(DEBUG_FLAG) $(DUMP_SYMBOLIC)
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

# make build_pass [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [DUMP_SYMBOLIC=-DFAF_DUMP_SYMBOLIC]
build_pass:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
		   --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
	$(OP) -S -emit-llvm select.c \
    	  -o select.ll
	$(OPT) -passes=error-analysis -S select.ll \
								  -o select_ac.ll

	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
           --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AmplificationFactor.h \
    $(OP) -S -emit-llvm selectUsingPhi.c \
          -o selectUsingPhi.ll
	$(OPT) -passes=error-analysis -S selectUsingPhi.ll \
           						  -o selectUsingPhi_ac.ll

run_pass1:
	$(LLI) select_ac.ll
run_pass2:
	$(LLI) selectUsingPhi_ac.ll

build_original:
	gcc select.c -lm -o select.out
	gcc selectUsingPhi.c -lm -o selectUsingPhi.out

clean:
	rm -rf .fAF_logs a.out select.ll  selectUsingPhi.ll ex1_ac.ll select.out selectUsingPhi.out
