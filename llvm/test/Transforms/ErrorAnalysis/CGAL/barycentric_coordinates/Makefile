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

RUN_TIME_LIB_FLAGS = -function-to-analyze compute_barycentric_coordinates \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: barycentric_coordinates_ac.ll
	$(LLI) barycentric_coordinates_ac.ll

barycentric_coordinates_ac.ll: barycentric_coordinates.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) barycentric_coordinates.ll -o barycentric_coordinates_ac.ll

barycentric_coordinates.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) barycentric_coordinates.c -o barycentric_coordinates.ll

build_original:
	gcc barycentric_coordinates.c -o barycentric_coordinates.out

clean:
	rm -rf barycentric_coordinates.ll barycentric_coordinates_ac.ll a.out barycentric_coordinates.out *.txt \
#	.fAF_logs
