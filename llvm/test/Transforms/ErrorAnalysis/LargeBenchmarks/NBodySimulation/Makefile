LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

OP =  -O1
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) -DMEMORY_OPT -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze run_simulation \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/Statistics.h

run: n_body_simulation_ac.ll
	$(LLI) n_body_simulation_ac.ll

n_body_simulation_ac.ll: n_body_simulation.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) n_body_simulation.ll -o n_body_simulation_ac.ll

n_body_simulation.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) n_body_simulation.c -o n_body_simulation.ll

build_original:
	gcc n_body_simulation.c -lm -o n_body_simulation.out

clean:
	rm -rf n_body_simulation.ll n_body_simulation_ac.ll .fAF_logs n_body_simulation_ac.out *.txt
