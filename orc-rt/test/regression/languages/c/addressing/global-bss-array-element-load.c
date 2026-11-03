// Check that JIT'd code can load the last element of a large global
// zero-initialized (BSS) array defined in the same object.
//
// Stresses: that BSS memory is allocated at its full size, which comes from
// the section header since BSS has no contents in the object file. The array
// is 1MB, larger than any common OS page size or allocation granularity (e.g.
// 16KB on Apple arm64, 64KB on some arm64 Linux kernels and for Windows
// allocations), so undersized memory (e.g. a single page) leaves the last
// element unmapped.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

#define NumElements (1024 * 1024 / sizeof(int))

// Explicitly zero-initialized, so that it can't become a common symbol. No
// barrier needed: non-const globals' values can't be propagated.
int Array[NumElements] = {0};

int main(void) {
  if (Array[NumElements - 1] != 0)
    return 1;
  return 0;
}
