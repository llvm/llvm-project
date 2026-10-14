// Check that JIT'd code can store to static data defined in the same object.
//
// Stresses: that the data section is writable, and fixups for code storing to
// data in the same object. Static data is always addressed directly, so the
// store itself carries a fixup, which may differ from the fixups used for loads
// (e.g. on RISC-V).
//
// RUN: %{cc} -O0 -c -o %t.O0.o %s
// RUN: %{obj-jit} -show-jit-result %t.O0.o | FileCheck %s
// RUN: %{cc} -O2 -c -o %t.O2.o %s
// RUN: %{obj-jit} -show-jit-result %t.O2.o | FileCheck %s

// CHECK: JIT result: 0

static int Data = 1;

int main(void) {
  Data = 42;
  // Volatile load, to prevent store-to-load forwarding.
  if (*(volatile int *)&Data != 42)
    return 1;
  return 0;
}
