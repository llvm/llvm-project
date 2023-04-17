LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

# -emit-pristine-llvm: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang8-emit-pristine-llvm
# -fno-slp-vectorize: https://llvm.org/docs/Vectorizers.html#the-slp-vectorizer
OP = -O1 -fno-slp-vectorize
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) $(MEMORY_OPT) -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze ex0 \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: Iterative_GramSchmidt_ac.ll
	$(LLI) Iterative_GramSchmidt_ac.ll

Iterative_GramSchmidt_ac.ll: Iterative_GramSchmidt.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) Iterative_GramSchmidt.ll -o Iterative_GramSchmidt_ac.ll

Iterative_GramSchmidt.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) Iterative_GramSchmidt.c -o Iterative_GramSchmidt.ll

build_original:
	gcc Iterative_GramSchmidt.c -o Iterative_GramSchmidt.out

clean:
	rm -rf Iterative_GramSchmidt.ll Iterative_GramSchmidt_ac.ll .fAF_logs a.out *.txt
