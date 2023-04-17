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

RUN_TIME_LIB_FLAGS = -function-to-analyze conjugateGradientSolver \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: cg_destructured_2_ac.ll
	$(LLI) cg_destructured_2_ac.ll

cg_destructured_2_ac.ll: cg_destructured_2.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) cg_destructured_2.ll -o cg_destructured_2_ac.ll

cg_destructured_2.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) cg_destructured_2.cpp -o cg_destructured_2.ll

build_original:
	g++ cg_destructured_2.cpp -o cg_destructured_2.out

clean:
	rm -rf cg_destructured_2.ll cg_destructured_2_ac.ll a.out cg_destructured_2.out *.txt \
#	.fAF_logs
