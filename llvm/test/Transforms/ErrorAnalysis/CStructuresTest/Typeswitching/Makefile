LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

# -emit-pristine-llvm: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang8-emit-pristine-llvm
OP = -emit-pristine-llvm
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) $(MEMORY_OPT) -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze implicit_type_casting \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: typeswitching_ac.ll
	$(LLI) typeswitching_ac.ll

typeswitching_ac.ll: typeswitching.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) typeswitching.ll -o typeswitching_ac.ll

typeswitching.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) typeswitching.c -o typeswitching.ll

build_original:
	gcc typeswitching.c -o typeswitching.out

clean:
	rm -rf typeswitching.ll typeswitching_ac.ll .fAF_logs a.out *.txt
