// RUN: llvm-mc -triple x86_64-unknown-unknown -mattr=+avx --show-encoding %s | FileCheck %s

// The VCMP comparison predicate may be a symbol. The VEX3-to-VEX2 shrink used
// to read it as a literal when deciding whether the operands commute, and
// assert.

// CHECK: vcmpps $f0, %xmm0, %xmm1, %xmm2
// CHECK-SAME: encoding: [0xc5,0xf0,0xc2,0xd0,A]
vcmpps $f0, %xmm0, %xmm1, %xmm2

// A literal predicate still commutes to reach the two-byte VEX prefix. Without
// the commute this would need the three-byte form.

// CHECK: vcmpeqps %xmm1, %xmm8, %xmm2
// CHECK-SAME: encoding: [0xc5,0xb8,0xc2,0xd1,0x00]
vcmpeqps %xmm8, %xmm1, %xmm2

// CHECK: vcmpltps %xmm8, %xmm1, %xmm2
// CHECK-SAME: encoding: [0xc4,0xc1,0x70,0xc2,0xd0,0x01]
vcmpps $1, %xmm8, %xmm1, %xmm2
