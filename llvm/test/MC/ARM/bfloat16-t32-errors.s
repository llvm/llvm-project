// RUN: not llvm-mc -triple thumbv8 -mattr=-bf16 < %s 2>&1 | FileCheck %s

vdot.bf16     d3, d4, d5
// CHECK: instruction requires: BFloat16 floating point extension
// CHECK-NEXT: vdot.bf16     d3, d4, d5

vdot.bf16    q0, q1, q2
// CHECK: instruction requires: BFloat16 floating point extension
// CHECK-NEXT: vdot.bf16    q0, q1, q2

vdot.bf16    d3, d4, d5[1]
// CHECK: instruction requires: BFloat16 floating point extension
// CHECK-NEXT: vdot.bf16    d3, d4, d5[1]

vdot.bf16    q0, q1, d5[1]
// CHECK: instruction requires: BFloat16 floating point extension
// CHECK-NEXT: vdot.bf16    q0, q1, d5[1]

vmmla.bf16  q0, q1, q2
// CHECK: instruction requires: BFloat16 floating point extension
// CHECK-NEXT: vmmla.bf16  q0, q1, q2

vcvt.bf16.f32 d1, q3
// CHECK: instruction requires: BFloat16 floating point extension
// CHECK-NEXT: vcvt.bf16.f32 d1, q3

vcvtbeq.bf16.f32  s1, s3
// CHECK: note: instruction requires: BFloat16 floating point extension
// CHECK-NEXT: vcvtbeq.bf16.f32  s1, s3
vcvttne.bf16.f32 s1, s3
// CHECK: note: instruction requires: BFloat16 floating point extension
// CHECK-NEXT: vcvttne.bf16.f32 s1, s3
