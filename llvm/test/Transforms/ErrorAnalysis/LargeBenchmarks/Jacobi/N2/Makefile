LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

# -emit-pristine-llvm: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang8-emit-pristine-llvm
# -fno-slp-vectorize: https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer
OP = -O1 -fno-slp-vectorize
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) $(MEMORYOPT) -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze solve_system \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: jacobi_n2_ac.ll
	$(LLI) jacobi_n2_ac.ll

jacobi_n2_ac.ll: jacobi_n2.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) jacobi_n2.ll -o jacobi_n2_ac.ll

jacobi_n2.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) jacobi_n2.c -o jacobi_n2.ll

build_original:
	gcc jacobi_n2.c -lm -o jacobi_n2.out

clean:
	rm -rf jacobi_n2.ll jacobi_n2_ac.ll .fAF_logs jacobi_n2_ac.out *.txt
