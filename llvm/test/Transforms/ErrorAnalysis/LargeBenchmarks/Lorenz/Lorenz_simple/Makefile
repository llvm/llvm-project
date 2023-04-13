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

RUN_TIME_LIB_FLAGS = -function-to-analyze solve_lorenz \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/Statistics.h

run: lorenz_simple_ac.ll
	$(LLI) lorenz_simple_ac.ll

lorenz_simple_ac.ll: lorenz_simple.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) lorenz_simple.ll -o lorenz_simple_ac.ll

lorenz_simple.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) lorenz_simple.c -o lorenz_simple.ll

build_original:
	gcc lorenz_simple.c -lm -o lorenz_simple.out

clean:
	rm -rf lorenz_simple.ll lorenz_simple_ac.ll a.out lorenz_simple.out *.txt \
#	.fAF_logs