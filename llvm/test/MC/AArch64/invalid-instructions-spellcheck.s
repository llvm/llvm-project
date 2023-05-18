// RUN: not llvm-mc -triple=aarch64 -show-encoding < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=aarch64 -mattr=-neon -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-NEON

// This tests the mnemonic spell checker.

// First check what happens when an instruction is omitted:

  w1, w2, w3

// CHECK:      error: unknown token in expression
// CHECK-NEXT: w1, w2, w3
// CHECK-NEXT:   ^
// CHECK-NEXT: error: invalid operand
// CHECK-NEXT: w1, w2, w3
// CHECK-NEXT:   ^

// We don't want to see a suggestion here; the edit distance is too large to
// give sensible suggestions:

  addddddddd w1, w2, w3

// CHECK:      error: unrecognized instruction mnemonic
// CHECK-NEXT: addddddddd w1, w2, w3
// CHECK-NEXT: ^

  addd w1, w2, w3

// CHECK:      error: unrecognized instruction mnemonic, did you mean: add, addp, adds, addv, fadd, madd?
// CHECK-NEXT: addd w1, w2, w3
// CHECK-NEXT: ^

// Instructions 'addv' and 'addp' are only available when NEON is enabled, so we
// don't want to see them here:

// CHECK-NO-NEON:      error: unrecognized instruction mnemonic, did you mean: add, adds, fadd, madd?
// CHECK-NO-NEON-NEXT: addd w1, w2, w3
// CHECK-NO-NEON-NEXT: ^
