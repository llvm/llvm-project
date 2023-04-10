LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

OP =  -O1
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) -DMEMORY_OPT -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze eulers_pendulum \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/Statistics.h

run: pendulum_ac.ll
	$(LLI) pendulum_ac.ll

pendulum_ac.ll: pendulum.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) pendulum.ll -o pendulum_ac.ll

pendulum.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) pendulum.c -o pendulum.ll

build_original:
	gcc pendulum.c -lm -o pendulum.out

clean:
	rm -rf pendulum.ll pendulum_ac.ll .fAF_logs pendulum_ac.out pendulum.out *.txt
