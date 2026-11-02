// Check that JIT'd code can load from static zero-initialized (BSS) data
// defined in the same object.
//
// Stresses: the BSS section's initial contents, and fixups for code addressing
// data in the same object. BSS has a size but no contents in the object file,
// so the JIT must allocate and zero its memory rather than copy it. (Fresh
// memory is usually zero already, so this test can't show that the JIT zeroes
// it.) References to static data may be relative to the section, rather than
// to the data's symbol.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

static int Data = 0;

int main(void) {
  // Volatile load, to prevent constant propagation of the initializer.
  if (*(volatile int *)&Data != 0)
    return 1;
  return 0;
}
