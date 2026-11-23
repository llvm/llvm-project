// Check that a pointer to a global function, stored in initialized data in the
// same object, can be called.
//
// Stresses: fixups for pointers stored in data, targeting functions in the
// same object.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

int addOne(int X) { return X + 1; }

// No barrier needed: a non-const global's value can't be propagated, so the
// call through AddOnePtr can't be devirtualized.
int (*AddOnePtr)(int) = addOne;

int main(void) {
  if (AddOnePtr(41) != 42)
    return 1;
  return 0;
}
