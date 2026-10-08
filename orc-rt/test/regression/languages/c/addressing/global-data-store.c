// Check that JIT'd code can store to global data defined in the same object.
//
// Stresses: fixups for code addressing data in the same object, and that the
// data is writable.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

int Data = 1;

int main(void) {
  Data = 42;
  // Volatile load, to prevent store-to-load forwarding.
  if (*(volatile int *)&Data != 42)
    return 1;
  return 0;
}
