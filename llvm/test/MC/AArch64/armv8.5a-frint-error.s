// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.5a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

// FP-to-int rounding, vector, illegal
frint32z v0.4h, v0.4h
frint32z v0.8b, v0.8b
frint32z v0.8h, v0.8h
frint32z v0.16b, v0.16b
frint64z v0.4h, v0.4h
frint64z v0.8b, v0.8b
frint64z v0.8h, v0.8h
frint64z v0.16b, v0.16b
frint32x v0.4h, v0.4h
frint32x v0.8b, v0.8b
frint32x v0.8h, v0.8h
frint32x v0.16b, v0.16b
frint64x v0.4h, v0.4h
frint64x v0.8b, v0.8b
frint64x v0.8h, v0.8h
frint64x v0.16b, v0.16b

// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint32z v0.4h, v0.4h
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint32z v0.8b, v0.8b
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint32z v0.8h, v0.8h
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint32z v0.16b, v0.16b
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint64z v0.4h, v0.4h
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint64z v0.8b, v0.8b
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint64z v0.8h, v0.8h
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint64z v0.16b, v0.16b
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint32x v0.4h, v0.4h
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint32x v0.8b, v0.8b
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint32x v0.8h, v0.8h
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint32x v0.16b, v0.16b
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint64x v0.4h, v0.4h
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint64x v0.8b, v0.8b
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint64x v0.8h, v0.8h
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR-NEXT: frint64x v0.16b, v0.16b
