// Check that JIT'd code can load an element of a static array defined in the
// same object, at an offset of more than a page from the start of the array.
//
// Stresses: fixups with addends for code addressing data in the same object,
// where the addend moves the target onto a different page from the symbol.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

static int Array[4096] = {[1500] = 42};

int main(void) {
  // Volatile load, to prevent constant propagation of the initializer.
  if (*(volatile int *)&Array[1500] != 42)
    return 1;
  return 0;
}
