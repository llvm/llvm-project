// Check that JIT'd code can load from global zero-initialized (BSS) data
// defined in the same object.
//
// Stresses: the BSS section's initial contents, and fixups for code addressing
// a global symbol in the same object. BSS has a size but no contents in the
// object file, so the JIT must allocate and zero its memory rather than copy
// it. (Fresh memory is usually zero already, so this test can't show that the
// JIT zeroes it.)
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

// Explicitly zero-initialized, so that it can't become a common symbol. No
// barrier needed: non-const globals' values can't be propagated.
int Data = 0;

int main(void) {
  if (Data != 0)
    return 1;
  return 0;
}
