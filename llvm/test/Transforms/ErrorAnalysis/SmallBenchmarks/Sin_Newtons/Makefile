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

RUN_TIME_LIB_FLAGS = -function-to-analyze newtons_sin \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/Statistics.h

run: sin_newtons_ac.ll
	$(LLI) sin_newtons_ac.ll

sin_newtons_ac.ll: sin_newtons.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) sin_newtons.ll -o sin_newtons_ac.ll

sin_newtons.ll:
	$(CXX) $(CXX_FLAGS) -DFAF_DEBUG=3 $(RUN_TIME_FILES) $(OP) sin_newtons.c -o sin_newtons.ll

build_original:
	gcc sin_newtons.c -lm -o sin_newtons.out

clean:
	rm -rf sin_newtons.ll sin_newtons_ac.ll .fAF_logs sin_newtons_ac.out sin_newtons.out *.txt
