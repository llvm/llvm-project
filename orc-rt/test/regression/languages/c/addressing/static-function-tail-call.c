// Check that JIT'd code can tail-call a static function defined in the same
// object.
//
// Stresses: fixups for tail calls (branches rather than calls) to functions in
// the same object.
//
// Only runs at -O2: an unoptimized build may not emit tail calls.
//
// RUN: %{cc} -O2 -c -o %t.o %s
// RUN: %{obj-jit} -show-jit-result %t.o | FileCheck %s

// CHECK: JIT result: 0

// noinline, to prevent inlining.
__attribute__((noinline)) static int addOne(int X) { return X + 1; }

// noinline, to prevent inlining into main, where the tail call would become a
// plain call.
__attribute__((noinline)) static int tailCallAddOne(int X) { return addOne(X); }

// Non-const global argument, to prevent interprocedural constant propagation.
int Arg = 41;

int main(void) {
  if (tailCallAddOne(Arg) != 42)
    return 1;
  return 0;
}
