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

RUN_TIME_LIB_FLAGS = -function-to-analyze runge_kutta \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: runge_kutta_4_ac.ll
	$(LLI) runge_kutta_4_ac.ll

runge_kutta_4_ac.ll: runge_kutta_4.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) runge_kutta_4.ll -o runge_kutta_4_ac.ll

runge_kutta_4.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) runge_kutta_4.c -o runge_kutta_4.ll

build_original:
	gcc runge_kutta_4.c -lm -o runge_kutta_4.out

clean:
	rm -rf runge_kutta_4.ll runge_kutta_4_ac.ll runge_kutta_4_ac.out runge_kutta_4.out *.txt
	#.fAF_logs
