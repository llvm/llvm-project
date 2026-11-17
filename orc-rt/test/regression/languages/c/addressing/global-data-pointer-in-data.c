// Check that a pointer to global data, stored in initialized data in the same
// object, points to that data.
//
// Stresses: fixups for pointers stored in data, targeting data in the same
// object.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

// No barrier needed: non-const globals' values can't be propagated.
int Data = 42;
int *DataPtr = &Data;

int main(void) {
  if (*DataPtr != 42)
    return 1;
  return 0;
}
