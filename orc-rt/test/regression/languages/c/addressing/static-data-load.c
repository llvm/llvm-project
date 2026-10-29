// Check that JIT'd code can load from static data defined in the same object.
//
// Stresses: the data section's initial contents, and fixups for code
// addressing data in the same object. References to static data may be
// relative to the section, rather than to the data's symbol.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

static int Data = 42;

int main(void) {
  // Volatile load, to prevent constant propagation of the initializer.
  if (*(volatile int *)&Data != 42)
    return 1;
  return 0;
}
