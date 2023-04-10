LLVM_PROJECT_ROOT = /home/tanmay/Documents/Tools/llvm-project
RUN_TIME_LIB_ROOT = /home/tanmay/Documents/Tools/llvm-project/llvm/include/llvm/Transforms/ErrorAnalysis

# -emit-pristine-llvm: https://clang.llvm.org/docs/ClangCommandLineReference.html#cmdoption-clang8-emit-pristine-llvm
OP = -O1
CXX = $(LLVM_PROJECT_ROOT)/build/bin/clang
# [DEBUG_FLAG=-DFAF_DEBUG<=debug level>] [MEMORYOPT=-DMEMORY_OPT]
CXX_FLAGS = $(DEBUG_FLAG) $(MEMORY_OPT) -g -S -emit-llvm -mllvm -disable-llvm-optzns

OPT = $(LLVM_PROJECT_ROOT)/build/bin/opt
OPT_FLAGS = -S -passes=error-analysis

LLI = $(LLVM_PROJECT_ROOT)/build/bin/lli

RUN_TIME_LIB_FLAGS = -function-to-analyze solve_lorenz \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/Statistics.h


run: lorenz_simple_16.out
	./lorenz_simple_16.out

lorenz_simple_16.out: lorenz_simple_16_ac.ll
	$(CXX) lorenz_simple_16_ac.ll -o lorenz_simple_16.out \
	-lm \
	/home/tanmay/Documents/Tools/cpfloat/build/lib/libcpfloat.so \
	-L/home/tanmay/Documents/Tools/cpfloat/deps/pcg-c/src \
	-lpcg_random

lorenz_simple_16_ac.ll: lorenz_simple_16.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) lorenz_simple_16.ll -o lorenz_simple_16_ac.ll

lorenz_simple_16.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) \
	-I/home/tanmay/Documents/Tools/cpfloat/src \
	-include /home/tanmay/Documents/Tools/cpfloat/deps/pcg-c/include/pcg_variants.h \
	lorenz_simple_16.c -o lorenz_simple_16.ll

build_original:
	gcc lorenz_simple_16.c -lm -o lorenz_simple_16.out

clean:
	rm -rf lorenz_simple_16.ll lorenz_simple_16_ac.ll lorenz_simple_16.out .fAF_logs a.out *.txt
