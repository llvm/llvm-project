// Check that JIT'd code can store to the last element of a large global
// zero-initialized (BSS) array defined in the same object.
//
// Stresses: that all of a 1MB BSS section, spanning many OS pages, is
// writable, not just its first page.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

#define NumElements (1024 * 1024 / sizeof(int))

// Explicitly zero-initialized, so that it can't become a common symbol.
int Array[NumElements] = {0};

int main(void) {
  Array[NumElements - 1] = 42;
  // Volatile load, to prevent store-to-load forwarding.
  if (*(volatile int *)&Array[NumElements - 1] != 42)
    return 1;
  return 0;
}
