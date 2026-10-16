// Check that JIT'd code can load from global data defined in the same object.
//
// Stresses: the data section's initial contents, and fixups for code
// addressing a global symbol in the same object.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

// No barrier needed: non-const globals' values can't be propagated.
int Data = 42;

int main(void) {
  if (Data != 42)
    return 1;
  return 0;
}
