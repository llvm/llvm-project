LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

# -emit-pristine-llvm: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang8-emit-pristine-llvm
# -fno-slp-vectorize: https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer
OP = -O1 -fno-slp-vectorize
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) -DMEMORY_OPT -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze newton_raphson \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: newton_raphson_ac.ll
	$(LLI) newton_raphson_ac.ll

newton_raphson_ac.ll: newton_raphson.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) newton_raphson.ll -o newton_raphson_ac.ll

newton_raphson.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) newton_raphson.c -o newton_raphson.ll

build_original:
	gcc newton_raphson.c -lm -o newton_raphson.out

clean:
	rm -rf newton_raphson.ll newton_raphson_ac.ll newton_raphson_ac.out newton_raphson.out *.txt \
#	.fAF_logs
