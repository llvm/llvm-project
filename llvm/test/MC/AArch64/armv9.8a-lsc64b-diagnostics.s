// RUN: not llvm-mc -triple=aarch64 -show-encoding -mattr=+lsc64b < %s 2>&1 \
// RUN:        | FileCheck %s

lda64b w0, [x13]
// CHECK:      error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-NEXT: lda64b w0, [x13]
// CHECK-NEXT:        ^

lda64b x23, [x13]
// CHECK:      error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-NEXT: lda64b x23, [x13]
// CHECK-NEXT:        ^

lda64b x0, [x13, #8]
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: lda64b x0, [x13, #8]
// CHECK-NEXT:             ^

stl64b w14, [x13]
// CHECK:      error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-NEXT: stl64b w14, [x13]
// CHECK-NEXT:        ^

stl64b x29, [x13]
// CHECK:      error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-NEXT: stl64b x29, [x13]
// CHECK-NEXT:        ^

stl64b x14, [x13, #8]
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: stl64b x14, [x13, #8]
// CHECK-NEXT:              ^

stl64bv w1, x20, [x13]
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: stl64bv w1, x20, [x13]
// CHECK-NEXT:         ^

stl64bv x1, w20, [x13]
// CHECK:      error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-NEXT: stl64bv x1, w20, [x13]
// CHECK-NEXT:             ^

stl64bv x1, x23, [x13]
// CHECK:      error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-NEXT: stl64bv x1, x23, [x13]
// CHECK-NEXT:             ^

stl64bv x1, x20, [x13, #0]
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: stl64bv x1, x20, [x13, #0]
// CHECK-NEXT:                  ^

stl64bv0 w1, x22, [x13]
// CHECK:      error: invalid operand for instruction
// CHECK-NEXT: stl64bv0 w1, x22, [x13]
// CHECK-NEXT:          ^

stl64bv0 x1, w22, [x13]
// CHECK:      error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-NEXT: stl64bv0 x1, w22, [x13]
// CHECK-NEXT:              ^

stl64bv0 x1, x29, [x13]
// CHECK:      error: expected an even-numbered x-register in the range [x0,x22]
// CHECK-NEXT: stl64bv0 x1, x29, [x13]
// CHECK-NEXT:              ^
