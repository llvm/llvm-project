@ RUN: llvm-mc -triple thumbv8 -mattr=+neon,+crypto -show-encoding < %s | FileCheck %s

aesd.8  q0, q1
@ CHECK: aesd.8  q0, q1         @ encoding: [0xb0,0xff,0x42,0x03]
aese.8  q0, q1
@ CHECK: aese.8 q0, q1          @ encoding: [0xb0,0xff,0x02,0x03]
aesimc.8  q0, q1
@ CHECK: aesimc.8 q0, q1        @ encoding: [0xb0,0xff,0xc2,0x03]
aesmc.8  q0, q1
@ CHECK: aesmc.8 q0, q1         @ encoding: [0xb0,0xff,0x82,0x03]

sha1h.32  q0, q1
@ CHECK: sha1h.32  q0, q1       @ encoding: [0xb9,0xff,0xc2,0x02]
sha1su1.32  q0, q1
@ CHECK: sha1su1.32 q0, q1      @ encoding: [0xba,0xff,0x82,0x03]
sha256su0.32  q0, q1
@ CHECK: sha256su0.32 q0, q1    @ encoding: [0xba,0xff,0xc2,0x03]

sha1c.32  q0, q1, q2
@ CHECK: sha1c.32  q0, q1, q2   @ encoding: [0x02,0xef,0x44,0x0c]
sha1m.32  q0, q1, q2
@ CHECK: sha1m.32  q0, q1, q2   @ encoding: [0x22,0xef,0x44,0x0c]
sha1p.32  q0, q1, q2
@ CHECK: sha1p.32 q0, q1, q2    @ encoding: [0x12,0xef,0x44,0x0c]
sha1su0.32  q0, q1, q2
@ CHECK: sha1su0.32  q0, q1, q2      @ encoding: [0x32,0xef,0x44,0x0c]
sha256h.32  q0, q1, q2
@ CHECK: sha256h.32  q0, q1, q2      @ encoding: [0x02,0xff,0x44,0x0c]
sha256h2.32  q0, q1, q2
@ CHECK: sha256h2.32 q0, q1, q2      @ encoding: [0x12,0xff,0x44,0x0c]
sha256su1.32  q0, q1, q2
@ CHECK: sha256su1.32 q0, q1, q2     @ encoding: [0x22,0xff,0x44,0x0c]

vmull.p64 q8, d16, d17
@ CHECK: vmull.p64  q8, d16, d17    @ encoding: [0xe0,0xef,0xa1,0x0e]
