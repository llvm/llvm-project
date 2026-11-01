// Check that JIT'd code can store to global zero-initialized (BSS) data
// defined in the same object.
//
// Stresses: that the BSS section is writable, and fixups for code storing to a
// global symbol in the same object. Global data may be addressed through the
// GOT, in which case the store itself carries no fixup.
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

// Explicitly zero-initialized, so that it can't become a common symbol.
int Data = 0;

int main(void) {
  Data = 42;
  // Volatile load, to prevent store-to-load forwarding.
  if (*(volatile int *)&Data != 42)
    return 1;
  return 0;
}
