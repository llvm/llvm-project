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

RUN_TIME_LIB_FLAGS = -function-to-analyze temp \
					-evaluations 0

RUN_TIME_FILES = --include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AtomicCondition.h \
				--include $(RUN_TIME_LIB_ROOT)/AtomicCondition/AmplificationFactor.h \
				--include $(RUN_TIME_LIB_ROOT)/Utilities/RunTimeUtilities.h

run: forLoop_ac.ll
	$(LLI) forLoop_ac.ll

forLoop_ac.ll: forLoop.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) forLoop.ll -o forLoop_ac.ll

forLoop.ll:
	$(CXX) $(CXX_FLAGS) $(RUN_TIME_FILES) $(OP) forLoop.c -o forLoop.ll

build_original:
	gcc forLoop.c -o forLoop.out

clean:
	rm -rf forLoop.ll forLoop_ac.ll .fAF_logs a.out *.txt

forLoop16: forLoop16.o
	gcc forLoop16.o -o forLoop16.out \
	-lm \
	-lcpfloat \
	-L/home/tanmay/Documents/Tools/cpfloat/build/lib \
	-L /home/tanmay/Documents/Tools/cpfloat/deps/pcg-c/src \
	-lpcg_random

forLoop16.o:
	gcc -c -I/home/tanmay/Documents/Tools/cpfloat/src \
	forLoop16.c -o forLoop16.o \
	-include /home/tanmay/Documents/Tools/cpfloat/deps/pcg-c/include/pcg_variants.h

forLoop16_clang: forLoop16_clang_ac.ll
	$(CXX) forLoop16_clang_ac.ll -o forLoop16.out \
	-lm \
	/home/tanmay/Documents/Tools/cpfloat/build/lib/libcpfloat.so \
	-L/home/tanmay/Documents/Tools/cpfloat/deps/pcg-c/src \
	-lpcg_random

forLoop16_clang_ac.ll: forLoop16_clang.ll
	$(OPT) $(OPT_FLAGS) $(RUN_TIME_LIB_FLAGS) forLoop16_clang.ll -o forLoop16_clang_ac.ll

forLoop16_clang.ll:
	$(CXX) $(CXX_FLAGS) $(OP) $(RUN_TIME_FILES) -I/home/tanmay/Documents/Tools/cpfloat/src \
	forLoop16.c -o forLoop16_clang.ll \
	-include /home/tanmay/Documents/Tools/cpfloat/deps/pcg-c/include/pcg_variants.h