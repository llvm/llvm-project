LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

OP =  -O1
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) $(MEMORY_OPT) -g -S -emit-llvm

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze compute_center \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: circleofpts_ac.ll
	$(LLI) circleofpts_ac.ll

circleofpts_ac.ll: circleofpts.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) circleofpts.ll -o circleofpts_ac.ll

circleofpts.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) circleofpts.cpp -o circleofpts.ll

build_original:
	gcc circleofpts.cpp -lm -o circleofpts.out

clean:
	rm -rf circleofpts.ll circleofpts_ac.ll a.out circleofpts.out *.txt \
	.fAF_logs
