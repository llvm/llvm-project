// RUN: llvm-mc -triple arm -mattr=+bf16,+neon -show-encoding < %s | FileCheck %s  --check-prefix=CHECK
// RUN: llvm-mc -triple arm -mattr=+v8.6a -show-encoding < %s | FileCheck %s  --check-prefix=CHECK

vdot.bf16     d3, d4, d5
// CHECK:     vdot.bf16  d3, d4, d5     @ encoding: [0x05,0x3d,0x04,0xfc]
vdot.bf16    q0, q1, q2
// CHECK-NEXT:     vdot.bf16 q0, q1, q2     @ encoding: [0x44,0x0d,0x02,0xfc]
vdot.bf16     d3, d4, d5[1]
// CHECK-NEXT:     vdot.bf16  d3, d4, d5[1] @ encoding: [0x25,0x3d,0x04,0xfe]
vdot.bf16    q0, q1, d5[1]
// CHECK-NEXT:     vdot.bf16  q0, q1, d5[1] @ encoding: [0x65,0x0d,0x02,0xfe]
vmmla.bf16  q0, q1, q2
// CHECK-NEXT:     vmmla.bf16 q0, q1, q2   @ encoding: [0x44,0x0c,0x02,0xfc]
vcvt.bf16.f32 d1, q3
// CHECK-NEXT:     vcvt.bf16.f32   d1, q3    @ encoding: [0x46,0x16,0xb6,0xf3]
vcvtbeq.bf16.f32  s1, s3
// CHECK-NEXT: vcvtbeq.bf16.f32 s1, s3       @ encoding: [0x61,0x09,0xf3,0x0e]
vcvttne.bf16.f32 s1, s3
// CHECK-NEXT: vcvttne.bf16.f32 s1, s3       @ encoding: [0xe1,0x09,0xf3,0x1e]
vfmat.bf16 q0, q0, q0
//CHECK-NEXT: vfmat.bf16      q0, q0, q0      @ encoding: [0x50,0x08,0x30,0xfc]
vfmat.bf16 q0, q0, q15
//CHECK-NEXT: vfmat.bf16      q0, q0, q15     @ encoding: [0x7e,0x08,0x30,0xfc]
vfmat.bf16 q0, q15, q0
//CHECK-NEXT: vfmat.bf16      q0, q15, q0     @ encoding: [0xd0,0x08,0x3e,0xfc]
vfmat.bf16 q0, q15, q15
//CHECK-NEXT: vfmat.bf16      q0, q15, q15     @ encoding: [0xfe,0x08,0x3e,0xfc]
vfmat.bf16 q7, q0, q0
//CHECK-NEXT: vfmat.bf16      q7, q0, q0      @ encoding: [0x50,0xe8,0x30,0xfc]
vfmat.bf16 q8, q0, q0
//CHECK-NEXT: vfmat.bf16      q8, q0, q0      @ encoding: [0x50,0x08,0x70,0xfc]
vfmab.bf16 q0, q0, q0
//CHECK-NEXT: vfmab.bf16      q0, q0, q0      @ encoding: [0x10,0x08,0x30,0xfc]
vfmab.bf16 q0, q0, q15
//CHECK-NEXT: vfmab.bf16      q0, q0, q15     @ encoding: [0x3e,0x08,0x30,0xfc]
vfmab.bf16 q0, q15, q0
//CHECK-NEXT: vfmab.bf16      q0, q15, q0     @ encoding: [0x90,0x08,0x3e,0xfc]
vfmab.bf16 q0, q15, q15
//CHECK-NEXT: vfmab.bf16      q0, q15, q15    @ encoding: [0xbe,0x08,0x3e,0xfc]
vfmab.bf16 q7, q0, q0
//CHECK-NEXT: vfmab.bf16      q7, q0, q0      @ encoding: [0x10,0xe8,0x30,0xfc]
vfmab.bf16 q8, q0, q0
//CHECK-NEXT: vfmab.bf16      q8, q0, q0      @ encoding: [0x10,0x08,0x70,0xfc]
vfmat.bf16 q0, q0, d0[0]
//CHECK-NEXT:  vfmat.bf16   q0, q0, d0[0]   @ encoding: [0x50,0x08,0x30,0xfe]
vfmat.bf16 q0, q0, d0[3]
//CHECK-NEXT:  vfmat.bf16   q0, q0, d0[3]   @ encoding: [0x78,0x08,0x30,0xfe]
vfmat.bf16 q0, q0, d7[0]
//CHECK-NEXT:  vfmat.bf16   q0, q0, d7[0]   @ encoding: [0x57,0x08,0x30,0xfe]
vfmab.bf16 q0, q0, d0[0]
//CHECK-NEXT:  vfmab.bf16   q0, q0, d0[0]   @ encoding: [0x10,0x08,0x30,0xfe]
vfmab.bf16 q0, q0, d0[3]
//CHECK-NEXT:  vfmab.bf16   q0, q0, d0[3]   @ encoding: [0x38,0x08,0x30,0xfe]
vfmab.bf16 q0, q0, d7[0]
//CHECK-NEXT:  vfmab.bf16   q0, q0, d7[0]   @ encoding: [0x17,0x08,0x30,0xfe]
