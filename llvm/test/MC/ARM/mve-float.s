# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s

# CHECK: vrintn.f16 q1, q0  @ encoding: [0xb6,0xff,0x40,0x24]
# CHECK-NOFP-NOT: vrintn.f16 q1, q0  @ encoding: [0xb6,0xff,0x40,0x24]
vrintn.f16 q1, q0

# CHECK: vrintn.f32 q0, q4  @ encoding: [0xba,0xff,0x48,0x04]
# CHECK-NOFP-NOT: vrintn.f32 q0, q4  @ encoding: [0xba,0xff,0x48,0x04]
vrintn.f32 q0, q4

# CHECK: vrinta.f16 q0, q1  @ encoding: [0xb6,0xff,0x42,0x05]
# CHECK-NOFP-NOT: vrinta.f16 q0, q1  @ encoding: [0xb6,0xff,0x42,0x05]
vrinta.f16 q0, q1

# CHECK: vrinta.f32 q1, q3  @ encoding: [0xba,0xff,0x46,0x25]
# CHECK-NOFP-NOT: vrinta.f32 q1, q3  @ encoding: [0xba,0xff,0x46,0x25]
vrinta.f32 q1, q3

# CHECK: vrintm.f16 q0, q5  @ encoding: [0xb6,0xff,0xca,0x06]
# CHECK-NOFP-NOT: vrintm.f16 q0, q5  @ encoding: [0xb6,0xff,0xca,0x06]
vrintm.f16 q0, q5

# CHECK: vrintm.f32 q0, q4  @ encoding: [0xba,0xff,0xc8,0x06]
# CHECK-NOFP-NOT: vrintm.f32 q0, q4  @ encoding: [0xba,0xff,0xc8,0x06]
vrintm.f32 q0, q4

# CHECK: vrintp.f16 q1, q0  @ encoding: [0xb6,0xff,0xc0,0x27]
# CHECK-NOFP-NOT: vrintp.f16 q1, q0  @ encoding: [0xb6,0xff,0xc0,0x27]
vrintp.f16 q1, q0

# CHECK: vrintp.f32 q0, q1  @ encoding: [0xba,0xff,0xc2,0x07]
# CHECK-NOFP-NOT: vrintp.f32 q0, q1  @ encoding: [0xba,0xff,0xc2,0x07]
vrintp.f32 q0, q1

# CHECK: vrintx.f16 q1, q2  @ encoding: [0xb6,0xff,0xc4,0x24]
# CHECK-NOFP-NOT: vrintx.f16 q1, q2  @ encoding: [0xb6,0xff,0xc4,0x24]
vrintx.f16 q1, q2

# CHECK: vrintx.f32 q1, q1  @ encoding: [0xba,0xff,0xc2,0x24]
# CHECK-NOFP-NOT: vrintx.f32 q1, q1  @ encoding: [0xba,0xff,0xc2,0x24]
vrintx.f32 q1, q1

# CHECK: vrintz.f16 q1, q6  @ encoding: [0xb6,0xff,0xcc,0x25]
# CHECK-NOFP-NOT: vrintz.f16 q1, q6  @ encoding: [0xb6,0xff,0xcc,0x25]
vrintz.f16 q1, q6

# CHECK: vrintz.f32 q1, q0  @ encoding: [0xba,0xff,0xc0,0x25]
# CHECK-NOFP-NOT: vrintz.f32 q1, q0  @ encoding: [0xba,0xff,0xc0,0x25]
vrintz.f32 q1, q0

# CHECK: vrintr.f32 s0, s1 @ encoding: [0xb6,0xee,0x60,0x0a]
# CHECK-NOFP-NOT: vrintr.f32 s0, s1 @ encoding: [0xb6,0xee,0x60,0x0a]
vrintr.f32.f32 s0, s1

# CHECK: vrintr.f64 d0, d1 @ encoding: [0xb6,0xee,0x41,0x0b]
# CHECK-NOFP-NOT: vrintr.f64 d0, d1 @ encoding: [0xb6,0xee,0x41,0x0b]
vrintr.f64.f64 d0, d1

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid instruction
vrintr.f32.f32 q0, q1

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid instruction
vrintr.f64 q0, q1

# CHECK: vmul.f16 q2, q1, q3  @ encoding: [0x12,0xff,0x56,0x4d]
# CHECK-NOFP-NOT: vmul.f16 q2, q1, q3  @ encoding: [0x12,0xff,0x56,0x4d]
vmul.f16 q2, q1, q3

# CHECK: vmul.f32 q0, q0, q5  @ encoding: [0x00,0xff,0x5a,0x0d]
# CHECK-NOFP-NOT: vmul.f32 q0, q0, q5  @ encoding: [0x00,0xff,0x5a,0x0d]
vmul.f32 q0, q0, q5

# CHECK: vcmla.f16 q3, q2, q1, #0  @ encoding: [0x24,0xfc,0x42,0x68]
# CHECK-NOFP-NOT: vcmla.f16 q3, q2, q1, #0  @ encoding: [0x24,0xfc,0x42,0x68]
vcmla.f16 q3, q2, q1, #0

# CHECK: vcmla.f16 q0, q0, q5, #90  @ encoding: [0xa0,0xfc,0x4a,0x08]
# CHECK-NOFP-NOT: vcmla.f16 q0, q0, q5, #90  @ encoding: [0xa0,0xfc,0x4a,0x08]
vcmla.f16 q0, q0, q5, #90

# CHECK: vcmla.f16 q3, q7, q2, #180  @ encoding: [0x2e,0xfd,0x44,0x68]
# CHECK-NOFP-NOT: vcmla.f16 q3, q7, q2, #180  @ encoding: [0x2e,0xfd,0x44,0x68]
vcmla.f16 q3, q7, q2, #180

# CHECK: vcmla.f16 q2, q7, q6, #270  @ encoding: [0xae,0xfd,0x4c,0x48]
# CHECK-NOFP-NOT: vcmla.f16 q2, q7, q6, #270  @ encoding: [0xae,0xfd,0x4c,0x48]
vcmla.f16 q2, q7, q6, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 0, 90, 180 or 270
vcmla.f16 q3, q2, q1, #200

# CHECK: vcmla.f32 q2, q6, q6, #0  @ encoding: [0x3c,0xfc,0x4c,0x48]
# CHECK-NOFP-NOT: vcmla.f32 q2, q6, q6, #0  @ encoding: [0x3c,0xfc,0x4c,0x48]
vcmla.f32 q2, q6, q6, #0

# CHECK: vcmla.f32 q7, q1, q3, #90  @ encoding: [0xb2,0xfc,0x46,0xe8]
# CHECK-NOFP-NOT: vcmla.f32 q7, q1, q3, #90  @ encoding: [0xb2,0xfc,0x46,0xe8]
vcmla.f32 q7, q1, q3, #90

# CHECK: vcmla.f32 q4, q5, q3, #180  @ encoding: [0x3a,0xfd,0x46,0x88]
# CHECK-NOFP-NOT: vcmla.f32 q4, q5, q3, #180  @ encoding: [0x3a,0xfd,0x46,0x88]
vcmla.f32 q4, q5, q3, #180

# CHECK: vcmla.f32 q3, q2, q7, #270  @ encoding: [0xb4,0xfd,0x4e,0x68]
# CHECK-NOFP-NOT: vcmla.f32 q3, q2, q7, #270  @ encoding: [0xb4,0xfd,0x4e,0x68]
vcmla.f32 q3, q2, q7, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 0, 90, 180 or 270
vcmla.f32 q3, q2, q1, #16

# CHECK: vfma.f16 q0, q2, q3  @ encoding: [0x14,0xef,0x56,0x0c]
# CHECK-NOFP-NOT: vfma.f16 q0, q2, q3  @ encoding: [0x14,0xef,0x56,0x0c]
vfma.f16 q0, q2, q3

# CHECK: vfma.f32 q0, q3, q7  @ encoding: [0x06,0xef,0x5e,0x0c]
# CHECK-NOFP-NOT: vfma.f32 q0, q3, q7  @ encoding: [0x06,0xef,0x5e,0x0c]
vfma.f32 q0, q3, q7

# CHECK: vfms.f16 q0, q2, q5  @ encoding: [0x34,0xef,0x5a,0x0c]
# CHECK-NOFP-NOT: vfms.f16 q0, q2, q5  @ encoding: [0x34,0xef,0x5a,0x0c]
vfms.f16 q0, q2, q5

# CHECK: vfms.f32 q1, q1, q2  @ encoding: [0x22,0xef,0x54,0x2c]
# CHECK-NOFP-NOT: vfms.f32 q1, q1, q2  @ encoding: [0x22,0xef,0x54,0x2c]
vfms.f32 q1, q1, q2

# CHECK: vadd.f16 q0, q0, q5  @ encoding: [0x10,0xef,0x4a,0x0d]
# CHECK-NOFP-NOT: vadd.f16 q0, q0, q5  @ encoding: [0x10,0xef,0x4a,0x0d]
vadd.f16 q0, q0, q5

# CHECK: vadd.f32 q1, q3, q0  @ encoding: [0x06,0xef,0x40,0x2d]
# CHECK-NOFP-NOT: vadd.f32 q1, q3, q0  @ encoding: [0x06,0xef,0x40,0x2d]
vadd.f32 q1, q3, q0

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid instruction
vaddeq.f32 q0, q1, q2

# CHECK:  vadd.f32  q0, q1, q2 @ encoding: [0x02,0xef,0x44,0x0d]
# CHECK-NOFP-NOT:  vadd.f32  q0, q1, q2 @ encoding: [0x02,0xef,0x44,0x0d]
vadd.f32 q0, q1, q2

# CHECK: vcadd.f16 q2, q1, q7, #90  @ encoding: [0x82,0xfc,0x4e,0x48]
# CHECK-NOFP-NOT: vcadd.f16 q2, q1, q7, #90  @ encoding: [0x82,0xfc,0x4e,0x48]
vcadd.f16 q2, q1, q7, #90

# CHECK: vcadd.f16 q2, q5, q7, #270  @ encoding: [0x8a,0xfd,0x4e,0x48]
# CHECK-NOFP-NOT: vcadd.f16 q2, q5, q7, #270  @ encoding: [0x8a,0xfd,0x4e,0x48]
vcadd.f16 q2, q5, q7, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 90 or 270
vcadd.f16 q2, q5, q7, #180

# CHECK: vcadd.f32 q0, q4, q7, #90  @ encoding: [0x98,0xfc,0x4e,0x08]
# CHECK-NOFP-NOT: vcadd.f32 q0, q4, q7, #90  @ encoding: [0x98,0xfc,0x4e,0x08]
vcadd.f32 q0, q4, q7, #90

# CHECK: vcadd.f32 q2, q2, q3, #270  @ encoding: [0x94,0xfd,0x46,0x48]
# CHECK-NOFP-NOT: vcadd.f32 q2, q2, q3, #270  @ encoding: [0x94,0xfd,0x46,0x48]
vcadd.f32 q2, q2, q3, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 90 or 270
vcadd.f32 q2, q5, q7, #0

# CHECK: vabd.f16 q0, q0, q6  @ encoding: [0x30,0xff,0x4c,0x0d]
# CHECK-NOFP-NOT: vabd.f16 q0, q0, q6  @ encoding: [0x30,0xff,0x4c,0x0d]
vabd.f16 q0, q0, q6

# CHECK: vabd.f32 q0, q1, q4  @ encoding: [0x22,0xff,0x48,0x0d]
# CHECK-NOFP-NOT: vabd.f32 q0, q1, q4  @ encoding: [0x22,0xff,0x48,0x0d]
vabd.f32 q0, q1, q4

# CHECK: vcvt.f16.s16 q1, q7, #1  @ encoding: [0xbf,0xef,0x5e,0x2c]
# CHECK-NOFP-NOT: vcvt.f16.s16 q1, q7, #1  @ encoding: [0xbf,0xef,0x5e,0x2c]
vcvt.f16.s16 q1, q7, #1

# CHECK: vcvt.f16.s16 q1, q7, #16  @ encoding: [0xb0,0xef,0x5e,0x2c]
# CHECK-NOFP-NOT: vcvt.f16.s16 q1, q7, #16  @ encoding: [0xb0,0xef,0x5e,0x2c]
vcvt.f16.s16 q1, q7, #16

# CHECK: vcvt.f16.s16 q1, q7, #11  @ encoding: [0xb5,0xef,0x5e,0x2c]
# CHECK-NOFP-NOT: vcvt.f16.s16 q1, q7, #11  @ encoding: [0xb5,0xef,0x5e,0x2c]
vcvt.f16.s16 q1, q7, #11

# CHECK: vcvt.s16.f16 q1, q1, #3  @ encoding: [0xbd,0xef,0x52,0x2d]
# CHECK-NOFP-NOT: vcvt.s16.f16 q1, q1, #3  @ encoding: [0xbd,0xef,0x52,0x2d]
vcvt.s16.f16 q1, q1, #3

# CHECK: vcvt.f16.u16 q2, q1, #10  @ encoding: [0xb6,0xff,0x52,0x4c]
# CHECK-NOFP-NOT: vcvt.f16.u16 q2, q1, #10  @ encoding: [0xb6,0xff,0x52,0x4c]
vcvt.f16.u16 q2, q1, #10

# CHECK: vcvt.u16.f16 q0, q0, #3  @ encoding: [0xbd,0xff,0x50,0x0d]
# CHECK-NOFP-NOT: vcvt.u16.f16 q0, q0, #3  @ encoding: [0xbd,0xff,0x50,0x0d]
vcvt.u16.f16 q0, q0, #3

# CHECK: vcvt.f32.s32 q1, q7, #1  @ encoding: [0xbf,0xef,0x5e,0x2e]
# CHECK-NOFP-NOT: vcvt.f32.s32 q1, q7, #1  @ encoding: [0xbf,0xef,0x5e,0x2e]
vcvt.f32.s32 q1, q7, #1

# CHECK: vcvt.f32.s32 q1, q7, #32  @ encoding: [0xa0,0xef,0x5e,0x2e]
# CHECK-NOFP-NOT: vcvt.f32.s32 q1, q7, #32  @ encoding: [0xa0,0xef,0x5e,0x2e]
vcvt.f32.s32 q1, q7, #32

# CHECK: vcvt.f32.s32 q1, q7, #6  @ encoding: [0xba,0xef,0x5e,0x2e]
# CHECK-NOFP-NOT: vcvt.f32.s32 q1, q7, #6  @ encoding: [0xba,0xef,0x5e,0x2e]
vcvt.f32.s32 q1, q7, #6

# CHECK: vcvt.s32.f32 q1, q0, #21  @ encoding: [0xab,0xef,0x50,0x2f]
# CHECK-NOFP-NOT: vcvt.s32.f32 q1, q0, #21  @ encoding: [0xab,0xef,0x50,0x2f]
vcvt.s32.f32 q1, q0, #21

# CHECK: vcvt.f32.u32 q1, q4, #4  @ encoding: [0xbc,0xff,0x58,0x2e]
# CHECK-NOFP-NOT: vcvt.f32.u32 q1, q4, #4  @ encoding: [0xbc,0xff,0x58,0x2e]
vcvt.f32.u32 q1, q4, #4

# CHECK: vcvt.u32.f32 q1, q5, #8  @ encoding: [0xb8,0xff,0x5a,0x2f]
# CHECK-NOFP-NOT: vcvt.u32.f32 q1, q5, #8  @ encoding: [0xb8,0xff,0x5a,0x2f]
vcvt.u32.f32 q1, q5, #8

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: MVE fixed-point immediate operand must be between 1 and 16
vcvt.f16.s16 q0, q1, #-1

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: MVE fixed-point immediate operand must be between 1 and 16
vcvt.f16.s16 q0, q1, #0

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: MVE fixed-point immediate operand must be between 1 and 16
vcvt.f16.s16 q0, q1, #17

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: MVE fixed-point immediate operand must be between 1 and 32
vcvt.f32.s32 q0, q1, #-1

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: MVE fixed-point immediate operand must be between 1 and 32
vcvt.f32.s32 q0, q1, #0

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: MVE fixed-point immediate operand must be between 1 and 32
vcvt.f32.s32 q0, q1, #33

# CHECK: vcvt.f16.s16 q0, q1  @ encoding: [0xb7,0xff,0x42,0x06]
# CHECK-NOFP-NOT: vcvt.f16.s16 q0, q1  @ encoding: [0xb7,0xff,0x42,0x06]
vcvt.f16.s16 q0, q1

# CHECK: vcvt.f16.u16 q0, q4  @ encoding: [0xb7,0xff,0xc8,0x06]
# CHECK-NOFP-NOT: vcvt.f16.u16 q0, q4  @ encoding: [0xb7,0xff,0xc8,0x06]
vcvt.f16.u16 q0, q4

# CHECK: vcvt.s16.f16 q0, q0  @ encoding: [0xb7,0xff,0x40,0x07]
# CHECK-NOFP-NOT: vcvt.s16.f16 q0, q0  @ encoding: [0xb7,0xff,0x40,0x07]
vcvt.s16.f16 q0, q0

# CHECK: vcvt.u16.f16 q0, q0  @ encoding: [0xb7,0xff,0xc0,0x07]
# CHECK-NOFP-NOT: vcvt.u16.f16 q0, q0  @ encoding: [0xb7,0xff,0xc0,0x07]
vcvt.u16.f16 q0, q0

# CHECK: vcvt.f32.s32 q0, q0  @ encoding: [0xbb,0xff,0x40,0x06]
# CHECK-NOFP-NOT: vcvt.f32.s32 q0, q0  @ encoding: [0xbb,0xff,0x40,0x06]
vcvt.f32.s32 q0, q0

# CHECK: vcvt.f32.u32 q0, q0  @ encoding: [0xbb,0xff,0xc0,0x06]
# CHECK-NOFP-NOT: vcvt.f32.u32 q0, q0  @ encoding: [0xbb,0xff,0xc0,0x06]
vcvt.f32.u32 q0, q0

# CHECK: vcvt.s32.f32 q0, q0  @ encoding: [0xbb,0xff,0x40,0x07]
# CHECK-NOFP-NOT: vcvt.s32.f32 q0, q0  @ encoding: [0xbb,0xff,0x40,0x07]
vcvt.s32.f32 q0, q0

# CHECK: vcvt.u32.f32 q0, q2  @ encoding: [0xbb,0xff,0xc4,0x07]
# CHECK-NOFP-NOT: vcvt.u32.f32 q0, q2  @ encoding: [0xbb,0xff,0xc4,0x07]
vcvt.u32.f32 q0, q2

# CHECK: vcvta.s16.f16 q0, q7  @ encoding: [0xb7,0xff,0x4e,0x00]
# CHECK-NOFP-NOT: vcvta.s16.f16 q0, q7  @ encoding: [0xb7,0xff,0x4e,0x00]
vcvta.s16.f16 q0, q7

# CHECK: vcvta.s32.f32 s2, s3 @ encoding: [0xbc,0xfe,0xe1,0x1a]
# CHECK-NOFP-NOT: vcvta.s32.f32 s2, s3 @ encoding: [0xbc,0xfe,0xe1,0x1a]
vcvta.s32.f32 s2, s3

# CHECK: vcvta.s16.f16 q0, q7 @ encoding: [0xb7,0xff,0x4e,0x00]
# CHECK-NOFP-NOT: vcvta.s16.f16 q0, q7 @ encoding: [0xb7,0xff,0x4e,0x00]
vcvta.s16.f16 q0, q7

# CHECK: vcvtn.u32.f32 q7, q6  @ encoding: [0xbb,0xff,0xcc,0xe1]
# CHECK-NOFP-NOT: vcvtn.u32.f32 q7, q6  @ encoding: [0xbb,0xff,0xcc,0xe1]
vcvtn.u32.f32 q7, q6

# CHECK: vcvtp.s32.f32 q0, q7  @ encoding: [0xbb,0xff,0x4e,0x02]
# CHECK-NOFP-NOT: vcvtp.s32.f32 q0, q7  @ encoding: [0xbb,0xff,0x4e,0x02]
vcvtp.s32.f32 q0, q7

# CHECK: vcvtm.u32.f32 q1, q4  @ encoding: [0xbb,0xff,0xc8,0x23]
# CHECK-NOFP-NOT: vcvtm.u32.f32 q1, q4  @ encoding: [0xbb,0xff,0xc8,0x23]
vcvtm.u32.f32 q1, q4

# CHECK: vneg.f16 q0, q7  @ encoding: [0xb5,0xff,0xce,0x07]
# CHECK-NOFP-NOT: vneg.f16 q0, q7  @ encoding: [0xb5,0xff,0xce,0x07]
vneg.f16 q0, q7

# CHECK: vneg.f32 q0, q2  @ encoding: [0xb9,0xff,0xc4,0x07]
# CHECK-NOFP-NOT: vneg.f32 q0, q2  @ encoding: [0xb9,0xff,0xc4,0x07]
vneg.f32 q0, q2

# CHECK: vabs.f16 q0, q2  @ encoding: [0xb5,0xff,0x44,0x07]
# CHECK-NOFP-NOT: vabs.f16 q0, q2  @ encoding: [0xb5,0xff,0x44,0x07]
vabs.f16 q0, q2

# CHECK: vabs.f32 q0, q0  @ encoding: [0xb9,0xff,0x40,0x07]
# CHECK-NOFP-NOT: vabs.f32 q0, q0  @ encoding: [0xb9,0xff,0x40,0x07]
vabs.f32 q0, q0

# CHECK: vmaxnma.f16 q1, q1  @ encoding: [0x3f,0xfe,0x83,0x2e]
# CHECK-NOFP-NOT: vmaxnma.f16 q1, q1  @ encoding: [0x3f,0xfe,0x83,0x2e]
vmaxnma.f16 q1, q1

# CHECK: vmaxnma.f32 q2, q6  @ encoding: [0x3f,0xee,0x8d,0x4e]
# CHECK-NOFP-NOT: vmaxnma.f32 q2, q6  @ encoding: [0x3f,0xee,0x8d,0x4e]
vmaxnma.f32 q2, q6

# CHECK: vminnma.f16 q0, q2  @ encoding: [0x3f,0xfe,0x85,0x1e]
# CHECK-NOFP-NOT: vminnma.f16 q0, q2  @ encoding: [0x3f,0xfe,0x85,0x1e]
vminnma.f16 q0, q2

# CHECK: vminnma.f32 q0, q1  @ encoding: [0x3f,0xee,0x83,0x1e]
# CHECK-NOFP-NOT: vminnma.f32 q0, q1  @ encoding: [0x3f,0xee,0x83,0x1e]
vminnma.f32 q0, q1

it eq
vaddeq.f32 s0, s1
# CHECK:  it  eq @ encoding: [0x08,0xbf]
# CHECK: vaddeq.f32  s0, s0, s1 @ encoding: [0x30,0xee,0x20,0x0a]
# CHECK-NOFP-NOT: vaddeq.f32  s0, s0, s1 @ encoding: [0x30,0xee,0x20,0x0a]

vpst
vaddt.f16 q0, q1, q2
# CHECK: vpst @ encoding: [0x71,0xfe,0x4d,0x0f]
# CHECK: vaddt.f16 q0, q1, q2 @ encoding: [0x12,0xef,0x44,0x0d]
# CHECK-NOFP-NOT: vaddt.f16 q0, q1, q2 @ encoding: [0x12,0xef,0x44,0x0d]

vpste
vcvtmt.u32.f32 q0, q1
vcvtne.s32.f32 q0, q1
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vtmt.u32.f32 q0, q1 @ encoding: [0xbb,0xff,0xc2,0x03]
# CHECK-NOFP-NOT: vtmt.u32.f32 q0, q1 @ encoding: [0xbb,0xff,0xc2,0x03]
# CHECK: vcvtne.s32.f32 q0, q1 @ encoding: [0xbb,0xff,0x42,0x01]
# CHECK-NOFP-NOT: vcvtne.s32.f32 q0, q1 @ encoding: [0xbb,0xff,0x42,0x01]

it ne
vcvtne.s32.f32 s0, s1
# CHECK: it ne @ encoding: [0x18,0xbf]
# CHECK: vcvtne.s32.f32 s0, s1 @ encoding: [0xbd,0xee,0xe0,0x0a]
# CHECK-NOFP-NOT: vcvtne.s32.f32 s0, s1 @ encoding: [0xbd,0xee,0xe0,0x0a]

it ge
vcvttge.f64.f16 d3, s1
# CHECK: it ge @ encoding: [0xa8,0xbf]
# CHECK: vcvttge.f64.f16 d3, s1 @ encoding: [0xb2,0xee,0xe0,0x3b]
# CHECK-NOFP-NOT: vcvttge.f64.f16 d3, s1 @ encoding: [0xb2,0xee,0xe0,0x3b]

# ----------------------------------------------------------------------
# The following tests have to go last because of the NOFP-NOT checks inside the
# VPT block.

vpte.f32 lt, q3, r1
vcvtt.u32.f32 q2, q0
vcvte.u32.f32 q1, q0
# CHECK: vpte.f32 lt, q3, r1      @ encoding: [0x77,0xee,0xc1,0x9f]
# CHECK-NOFP-NOT: vpte.f32 lt, q3, r1      @ encoding: [0x77,0xee,0xe1,0x8f]
# CHECK: vcvtt.u32.f32 q2, q0          @ encoding: [0xbb,0xff,0xc0,0x47]
# CHECK-NOFP-NOT: vcvtt.u32.f32 q2, q0          @ encoding: [0xbb,0xff,0xc0,0x47]
# CHECK: vcvte.u32.f32 q1, q0          @ encoding: [0xbb,0xff,0xc0,0x27]
# CHECK-NOFP-NOT: vcvte.u32.f32 q1, q0          @ encoding: [0xbb,0xff,0xc0,0x27]

ite eq
vcvteq.u32.f32 s0, s1
vcvtne.f32.u32 s0, s1
# CHECK: ite eq                      @ encoding: [0x0c,0xbf]
# CHECK: vcvteq.u32.f32 s0, s1          @ encoding: [0xbc,0xee,0xe0,0x0a]
# CHECK-NOFP-NOT: vcvteq.u32.f32 s0, s1          @ encoding: [0xbc,0xee,0xe0,0x0a]
# CHECK: vcvtne.f32.u32  s0, s1          @ encoding: [0xb8,0xee,0x60,0x0a]
# CHECK-NOFP-NOT: vcvtne.f32.u32  s0, s1          @ encoding: [0xb8,0xee,0x60,0x0a]

vpste
vmult.f16 q0, q1, q2
vmule.f16 q0, q1, q2
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vmult.f16 q0, q1, q2 @ encoding: [0x12,0xff,0x54,0x0d]
# CHECK-NOFP-NOT: vmult.f16 q0, q1, q2 @ encoding: [0x12,0xff,0x54,0x0d]
# CHECK: vmule.f16 q0, q1, q2 @ encoding: [0x12,0xff,0x54,0x0d]
# CHECK-NOFP-NOT: vmule.f16 q0, q1, q2 @ encoding: [0x12,0xff,0x54,0x0d]

ite eq
vmuleq.f64 d0, d0, d1
vmulne.f64 d1, d0, d2
# CHECK: ite eq @ encoding: [0x0c,0xbf]
# CHECK: vmuleq.f64 d0, d0, d1 @ encoding: [0x20,0xee,0x01,0x0b]
# CHECK-NOFP-NOT: vmuleq.f64 d0, d0, d1 @ encoding: [0x20,0xee,0x01,0x0b]
# CHECK: vmulne.f64 d1, d0, d2 @ encoding: [0x20,0xee,0x02,0x1b]
# CHECK-NOFP-NOT: vmulne.f64 d1, d0, d2 @ encoding: [0x20,0xee,0x02,0x1b]

it eq
vnegeq.f32 s0, s1
# CHECK: it eq @ encoding: [0x08,0xbf]
# CHECK: vnegeq.f32 s0, s1 @ encoding: [0xb1,0xee,0x60,0x0a]
# CHECK-NOFP-NOT: vnegeq.f32 s0, s1 @ encoding: [0xb1,0xee,0x60,0x0a]

itt eq
vnmuleq.f32 s0, s1, s2
vmuleq.f32 s0, s1, s2
# CHECK: itt eq @ encoding: [0x04,0xbf]
# CHECK: vnmuleq.f32 s0, s1, s2 @ encoding: [0x20,0xee,0xc1,0x0a]
# CHECK-NOFP-NOT: vnmuleq.f32 s0, s1, s2 @ encoding: [0x20,0xee,0xc1,0x0a]
# CHECK: vmuleq.f32 s0, s1, s2 @ encoding: [0x20,0xee,0x81,0x0a]
# CHECK-NOFP-NOT: vmuleq.f32 s0, s1, s2 @ encoding: [0x20,0xee,0x81,0x0a]

vpste
vrintnt.f16 q0, q1
vrintne.f32 q0, q1
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vrintnt.f16 q0, q1 @ encoding: [0xb6,0xff,0x42,0x04]
# CHECK-NOFP-NOT: vrintnt.f16 q0, q1 @ encoding: [0xb6,0xff,0x42,0x04]
# CHECK: vrintne.f32 q0, q1 @ encoding: [0xba,0xff,0x42,0x04]
# CHECK-NOFP-NOT: vrintne.f32 q0, q1 @ encoding: [0xba,0xff,0x42,0x04]
