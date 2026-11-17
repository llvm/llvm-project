// Check that JIT'd code can call a static function defined in the same object.
//
// Stresses: fixups for calls to functions in the same object.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

// noinline, to prevent inlining.
__attribute__((noinline)) static int addOne(int X) { return X + 1; }

// Non-const global argument, to prevent interprocedural constant propagation.
int Arg = 41;

int main(void) {
  if (addOne(Arg) != 42)
    return 1;
  return 0;
}
