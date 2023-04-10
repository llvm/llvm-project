LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

# -emit-pristine-llvm: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang8-emit-pristine-llvm
# -fno-slp-vectorize: https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer
OP = -O1 -fno-slp-vectorize
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
CXX_FLAGS = $(DEBUG_FLAG) $(MEMORY_OPT) -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze lu_factorization \
					-evaluations 10

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/Statistics.h

run: lu_factorization_ac.ll
	$(LLI) lu_factorization_ac.ll

lu_factorization_ac.ll: lu_factorization.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) lu_factorization.ll -o lu_factorization_ac.ll

lu_factorization.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) lu_factorization.c -o lu_factorization.ll

build_original:
	gcc lu_factorization.c -lm -o lu_factorization.out

clean:
	rm -rf lu_factorization.ll lu_factorization_ac.ll .fAF_logs a.out *.txt
