// RUN: llvm-mc -triple=aarch64-none-linux-gnu < %s | FileCheck %s
// RUN: not llvm-mc -mattr=+no-neg-immediates -triple=aarch64-none-linux-gnu < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-NEG-IMM

// CHECK: and x0, x1, #0xfffffffffffffffd
// CHECK: and x0, x1, #0xfffffffffffffffd
// CHECK-NO-NEG-IMM: instruction requires: NegativeImmediates
        and x0, x1, #~2
        bic x0, x1, #2

// CHECK: and w0, w1, #0xfffffffd
// CHECK: and w0, w1, #0xfffffffd
// CHECK-NO-NEG-IMM: instruction requires: NegativeImmediates
        and w0, w1, #~2
        bic w0, w1, #2

// CHECK: ands x0, x1, #0xfffffffffffffffd
// CHECK: ands x0, x1, #0xfffffffffffffffd
// CHECK-NO-NEG-IMM: instruction requires: NegativeImmediates
        ands x0, x1, #~2
        bics x0, x1, #2

// CHECK: ands w0, w1, #0xfffffffd
// CHECK: ands w0, w1, #0xfffffffd
// CHECK-NO-NEG-IMM: instruction requires: NegativeImmediates
        ands w0, w1, #~2
        bics w0, w1, #2

// CHECK: orr x0, x1, #0xfffffffffffffffd
// CHECK: orr x0, x1, #0xfffffffffffffffd
// CHECK-NO-NEG-IMM: instruction requires: NegativeImmediates
        orr x0, x1, #~2
        orn x0, x1, #2

// CHECK: orr w2, w1, #0xfffffffc
// CHECK: orr w2, w1, #0xfffffffc
// CHECK-NO-NEG-IMM: instruction requires: NegativeImmediates
        orr w2, w1, #~3
        orn w2, w1, #3

// CHECK: eor x0, x1, #0xfffffffffffffffd
// CHECK: eor x0, x1, #0xfffffffffffffffd
// CHECK-NO-NEG-IMM: instruction requires: NegativeImmediates
        eor x0, x1, #~2
        eon x0, x1, #2

// CHECK: eor w2, w1, #0xfffffffc
// CHECK: eor w2, w1, #0xfffffffc
// CHECK-NO-NEG-IMM: instruction requires: NegativeImmediates
        eor w2, w1, #~3
        eon w2, w1, #3
