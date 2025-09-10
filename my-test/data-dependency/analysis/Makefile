CC = clang
CFLAGS = -O0 -emit-llvm -S
LLVM_OPT = opt
LLVM_LLC = llc

# 生成所有 LLVM IR
all: $(patsubst %.c,%.ll,$(wildcard *.c))

%.ll: %.c
	$(CC) $(CFLAGS) $< -o $@

# 依賴性分析
analyze: all
	@echo "=== Flow Dependencies (RAW) ==="
	$(LLVM_OPT) -passes=print-memoryssa dep.ll -disable-output 2>&1 | head -20
	@echo "=== Loop Analysis ==="
	$(LLVM_OPT) -passes=print-loops dep.ll -disable-output 2>&1 | head -10

# 向量化分析
vectorize-analysis:
	$(LLVM_OPT) -passes=loop-vectorize -debug vectorizable.ll -S -o /dev/null

clean:
	rm -f *.ll *.s
