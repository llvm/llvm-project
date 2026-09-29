@ RUN: llvm-mc -triple armv8 -mattr=+neon,+crypto -show-encoding < %s | FileCheck %s
@ RUN: not llvm-mc -triple=armv7 -show-encoding < %s 2>&1 | FileCheck %s --check-prefix=CHECK-V7

aesd.8  q0, q1
aese.8  q0, q1
aesimc.8  q0, q1
aesmc.8  q0, q1
@ CHECK: aesd.8 q0, q1          @ encoding: [0x42,0x03,0xb0,0xf3]
@ CHECK: aese.8 q0, q1          @ encoding: [0x02,0x03,0xb0,0xf3]
@ CHECK: aesimc.8 q0, q1        @ encoding: [0xc2,0x03,0xb0,0xf3]
@ CHECK: aesmc.8 q0, q1         @ encoding: [0x82,0x03,0xb0,0xf3]
@ CHECK-V7: instruction requires: aes armv8
@ CHECK-V7: instruction requires: aes armv8
@ CHECK-V7: instruction requires: aes armv8
@ CHECK-V7: instruction requires: aes armv8

sha1h.32  q0, q1
sha1su1.32  q0, q1
sha256su0.32  q0, q1
@ CHECK: sha1h.32  q0, q1       @ encoding: [0xc2,0x02,0xb9,0xf3]
@ CHECK: sha1su1.32 q0, q1      @ encoding: [0x82,0x03,0xba,0xf3]
@ CHECK: sha256su0.32 q0, q1    @ encoding: [0xc2,0x03,0xba,0xf3]
@ CHECK-V7: instruction requires: sha2 armv8
@ CHECK-V7: instruction requires: sha2 armv8
@ CHECK-V7: instruction requires: sha2 armv8

sha1c.32  q0, q1, q2
sha1m.32  q0, q1, q2
sha1p.32  q0, q1, q2
sha1su0.32  q0, q1, q2
sha256h.32  q0, q1, q2
sha256h2.32  q0, q1, q2
sha256su1.32  q0, q1, q2
@ CHECK: sha1c.32  q0, q1, q2   @ encoding: [0x44,0x0c,0x02,0xf2]
@ CHECK: sha1m.32  q0, q1, q2   @ encoding: [0x44,0x0c,0x22,0xf2]
@ CHECK: sha1p.32 q0, q1, q2    @ encoding: [0x44,0x0c,0x12,0xf2]
@ CHECK: sha1su0.32  q0, q1, q2      @ encoding: [0x44,0x0c,0x32,0xf2]
@ CHECK: sha256h.32  q0, q1, q2      @ encoding: [0x44,0x0c,0x02,0xf3]
@ CHECK: sha256h2.32 q0, q1, q2      @ encoding: [0x44,0x0c,0x12,0xf3]
@ CHECK: sha256su1.32 q0, q1, q2     @ encoding: [0x44,0x0c,0x22,0xf3]
@ CHECK-V7: instruction requires: sha2 armv8
@ CHECK-V7: instruction requires: sha2 armv8
@ CHECK-V7: instruction requires: sha2 armv8
@ CHECK-V7: instruction requires: sha2 armv8
@ CHECK-V7: instruction requires: sha2 armv8
@ CHECK-V7: instruction requires: sha2 armv8
@ CHECK-V7: instruction requires: sha2 armv8

vmull.p64 q8, d16, d17
@ CHECK: vmull.p64  q8, d16, d17    @ encoding: [0xa1,0x0e,0xe0,0xf2]
@ CHECK-V7: instruction requires: aes armv8
