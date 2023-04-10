LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

OP =  -O1
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) $(MEMORYOPT) -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze solve_system \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/Statistics.h

run: jacobi_n4_ac.ll
	$(LLI) jacobi_n4_ac.ll

jacobi_n4_ac.ll: jacobi_n4.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) jacobi_n4.ll -o jacobi_n4_ac.ll

jacobi_n4.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) jacobi_n4.c -o jacobi_n4.ll

build_original:
	gcc jacobi_n4.c -lm -o jacobi_n4.out

clean:
	rm -rf jacobi_n4.ll jacobi_n4_ac.ll .fAF_logs jacobi_n4_ac.out *.txt
