LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

OP =  -O1
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) $(MEMORY_OPT) -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze solve_lorenz \
					-evaluations 10

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: lorenz_ode_ac.ll
	$(LLI) lorenz_ode_ac.ll

lorenz_ode_ac.ll: lorenz_ode.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) lorenz_ode.ll -o lorenz_ode_ac.ll

lorenz_ode.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) lorenz_ode.c -o lorenz_ode.ll

build_original:
	gcc lorenz_ode.c -lm -o lorenz_ode.out

clean:
	rm -rf lorenz_ode.ll lorenz_ode_ac.ll .fAF_logs a.out *.txt
