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

RUN_TIME_LIB_FLAGS = -function-to-analyze calculate_using_operands \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: IfConditional_ac.ll
	$(LLI) IfConditional_ac.ll

IfConditional_ac.ll: IfConditional.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) IfConditional.ll -o IfConditional_ac.ll

IfConditional.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) IfConditional.c -o IfConditional.ll

build_original:
	gcc IfConditional.c -o IfConditional.out

clean:
	rm -rf IfConditional.ll IfConditional_ac.ll .fAF_logs a.out *.txt
