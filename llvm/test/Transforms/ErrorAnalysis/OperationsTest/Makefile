OP = 	-O1
CXX = /home/tanmay/Documents/Tools/llvm-project/build/bin/clang
OPT = /home/tanmay/Documents/Tools/llvm-project/build/bin/opt
LLI = /home/tanmay/Documents/Tools/llvm-project/build/bin/lli

all:
	$(CXX) --include /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis/AtomicCondition/AtomicCondition.h \
	$(OP) -S -emit-llvm operations.c \
    	  -o operations.ll
	$(OPT) -passes=error-analysis -S operations.ll \
								  -o operations_ac.ll

run:
	$(LLI) operations_ac.ll

clean:
	rm -rf operations.ll operations_ac.ll .fAC_logs
