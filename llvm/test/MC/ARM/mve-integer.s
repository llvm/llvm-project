# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN: FileCheck --check-prefix=ERROR %s < %t

# CHECK: vmov.i32 q0, #0x1bff  @ encoding: [0x81,0xef,0x5b,0x0c]
vmov.i32 q0, #0x1bff

# CHECK: vmov.i16 q0, #0x5c  @ encoding: [0x85,0xef,0x5c,0x08]
vmov.i16 q0, #0x5c

# CHECK: vmov.i8 q0, #0x4c  @ encoding: [0x84,0xef,0x5c,0x0e]
vmov.i8 q0, #0x4c

# CHECK: vmov.f32 q0, #-3.625000e+00  @ encoding: [0x80,0xff,0x5d,0x0f]
vmov.f32 q0, #-3.625000e+00

# CHECK: vmov.f32 q0, #1.250000e-01  @ encoding: [0x84,0xef,0x50,0x0f]
vmov.f32 q0, #0.125

# CHECK: vmov.f32 q0, #1.328125e-01  @ encoding: [0x84,0xef,0x51,0x0f]
vmov.f32 q0, #0.1328125

# CHECK: vmov.f32 q0, #3.100000e+01  @ encoding: [0x83,0xef,0x5f,0x0f]
vmov.f32 q0, #31.0

# CHECK: vmov.f32 s16, s1  @ encoding: [0xb0,0xee,0x60,0x8a]
vmov.f32 s16, s1

# CHECK: vmov.f64 d0, d1  @ encoding: [0xb0,0xee,0x41,0x0b]
vmov.f64 d0, d1

# CHECK: vmov.i64 q0, #0xff0000ffffffffff  @ encoding: [0x81,0xff,0x7f,0x0e]
vmov.i64 q0, #0xff0000ffffffffff

# ERROR: [[@LINE+1]]:14: error: invalid operand for instruction
vmov.i32 q0, #0xabcd

# ERROR: [[@LINE+1]]:14: error: invalid operand for instruction
vmov.i16 q0, #0xabcd

# ERROR: [[@LINE+1]]:14: error: invalid operand for instruction
vmov.i32 q0, #0xabffffff

# ERROR: [[@LINE+1]]:14: error: invalid operand for instruction
vmov.i32 q0, #0xabffffff

# ERROR: [[@LINE+1]]:14: error: invalid operand for instruction
vmov.f32 q0, #0.0625

# ERROR: [[@LINE+1]]:14: error: invalid operand for instruction
vmov.f32 q0, #33.0

# CHECK: vmul.i8 q0, q0, q3  @ encoding: [0x00,0xef,0x56,0x09]
vmul.i8 q0, q0, q3

# CHECK: vmul.i16 q6, q0, q3  @ encoding: [0x10,0xef,0x56,0xc9]
vmul.i16 q6, q0, q3

# CHECK: vmul.i32 q7, q3, q6  @ encoding: [0x26,0xef,0x5c,0xe9]
vmul.i32 q7, q3, q6

# CHECK: vqrdmulh.s8 q0, q5, q5  @ encoding: [0x0a,0xff,0x4a,0x0b]
vqrdmulh.s8 q0, q5, q5

# CHECK: vqrdmulh.s16 q1, q4, q2  @ encoding: [0x18,0xff,0x44,0x2b]
vqrdmulh.s16 q1, q4, q2

# CHECK: vqrdmulh.s32 q0, q5, q0  @ encoding: [0x2a,0xff,0x40,0x0b]
vqrdmulh.s32 q0, q5, q0

# CHECK: vqdmulh.s8 q0, q4, q5  @ encoding: [0x08,0xef,0x4a,0x0b]
vqdmulh.s8 q0, q4, q5

# CHECK: vqdmulh.s16 q6, q4, q0  @ encoding: [0x18,0xef,0x40,0xcb]
vqdmulh.s16 q6, q4, q0

# CHECK: vqdmulh.s32 q5, q0, q6  @ encoding: [0x20,0xef,0x4c,0xab]
vqdmulh.s32 q5, q0, q6

# CHECK: vsub.i8 q3, q2, q5  @ encoding: [0x04,0xff,0x4a,0x68]
vsub.i8 q3, q2, q5

# CHECK: vsub.i16 q0, q3, q6  @ encoding: [0x16,0xff,0x4c,0x08]
vsub.i16 q0, q3, q6

# CHECK: vsub.i32 q0, q0, q6  @ encoding: [0x20,0xff,0x4c,0x08]
vsub.i32 q0, q0, q6

# CHECK: vadd.i8 q0, q2, q2  @ encoding: [0x04,0xef,0x44,0x08]
vadd.i8 q0, q2, q2

# CHECK: vadd.i16 q2, q2, q1  @ encoding: [0x14,0xef,0x42,0x48]
vadd.i16 q2, q2, q1

# CHECK: vadd.i32 q0, q0, q6  @ encoding: [0x20,0xef,0x4c,0x08]
vadd.i32 q0, q0, q6

# CHECK: vqsub.s8 q1, q6, q0  @ encoding: [0x0c,0xef,0x50,0x22]
vqsub.s8 q1, q6, q0

# CHECK: vqsub.s16 q0, q6, q1  @ encoding: [0x1c,0xef,0x52,0x02]
vqsub.s16 q0, q6, q1

# CHECK: vqsub.s32 q0, q0, q5  @ encoding: [0x20,0xef,0x5a,0x02]
vqsub.s32 q0, q0, q5

# CHECK: vqsub.u8 q0, q2, q6  @ encoding: [0x04,0xff,0x5c,0x02]
vqsub.u8 q0, q2, q6

# CHECK: vqsub.u16 q0, q7, q1  @ encoding: [0x1e,0xff,0x52,0x02]
vqsub.u16 q0, q7, q1

# CHECK: vqsub.u32 q1, q4, q7  @ encoding: [0x28,0xff,0x5e,0x22]
vqsub.u32 q1, q4, q7

# CHECK: vqadd.s8 q0, q1, q2 @ encoding: [0x02,0xef,0x54,0x00]
vqadd.s8 q0, q1, q2

# CHECK: vqadd.s8 q0, q4, q6  @ encoding: [0x08,0xef,0x5c,0x00]
vqadd.s8 q0, q4, q6

# CHECK: vqadd.s16 q0, q5, q5  @ encoding: [0x1a,0xef,0x5a,0x00]
vqadd.s16 q0, q5, q5

# CHECK: vqadd.s32 q0, q0, q4  @ encoding: [0x20,0xef,0x58,0x00]
vqadd.s32 q0, q0, q4

# CHECK: vqadd.u8 q0, q4, q2  @ encoding: [0x08,0xff,0x54,0x00]
vqadd.u8 q0, q4, q2

# CHECK: vqadd.u16 q4, q6, q6  @ encoding: [0x1c,0xff,0x5c,0x80]
vqadd.u16 q4, q6, q6

# CHECK: vqadd.u32 q0, q1, q2  @ encoding: [0x22,0xff,0x54,0x00]
vqadd.u32 q0, q1, q2

# CHECK: vabd.s8 q0, q0, q2  @ encoding: [0x00,0xef,0x44,0x07]
vabd.s8 q0, q0, q2

# CHECK: vabd.s16 q1, q5, q4  @ encoding: [0x1a,0xef,0x48,0x27]
vabd.s16 q1, q5, q4

# CHECK: vabd.s32 q2, q3, q2  @ encoding: [0x26,0xef,0x44,0x47]
vabd.s32 q2, q3, q2

# CHECK: vabd.u8 q1, q6, q4  @ encoding: [0x0c,0xff,0x48,0x27]
vabd.u8 q1, q6, q4

# CHECK: vabd.u16 q0, q6, q2  @ encoding: [0x1c,0xff,0x44,0x07]
vabd.u16 q0, q6, q2

# CHECK: vabd.u32 q0, q7, q4  @ encoding: [0x2e,0xff,0x48,0x07]
vabd.u32 q0, q7, q4

# CHECK: vrhadd.s8 q0, q1, q1  @ encoding: [0x02,0xef,0x42,0x01]
vrhadd.s8 q0, q1, q1

# CHECK: vrhadd.s16 q0, q1, q0  @ encoding: [0x12,0xef,0x40,0x01]
vrhadd.s16 q0, q1, q0

# CHECK: vrhadd.s32 q0, q4, q1  @ encoding: [0x28,0xef,0x42,0x01]
vrhadd.s32 q0, q4, q1

# CHECK: vrhadd.u8 q1, q0, q6  @ encoding: [0x00,0xff,0x4c,0x21]
vrhadd.u8 q1, q0, q6

# CHECK: vrhadd.u16 q2, q2, q5  @ encoding: [0x14,0xff,0x4a,0x41]
vrhadd.u16 q2, q2, q5

# CHECK: vrhadd.u32 q2, q3, q0  @ encoding: [0x26,0xff,0x40,0x41]
vrhadd.u32 q2, q3, q0

# CHECK: vhsub.s8 q0, q0, q2  @ encoding: [0x00,0xef,0x44,0x02]
vhsub.s8 q0, q0, q2

# CHECK: vhsub.s16 q1, q3, q1  @ encoding: [0x16,0xef,0x42,0x22]
vhsub.s16 q1, q3, q1

# CHECK: vhsub.s32 q0, q2, q5  @ encoding: [0x24,0xef,0x4a,0x02]
vhsub.s32 q0, q2, q5

# CHECK: vhsub.u8 q0, q4, q2  @ encoding: [0x08,0xff,0x44,0x02]
vhsub.u8 q0, q4, q2

# CHECK: vhsub.u16 q0, q7, q5  @ encoding: [0x1e,0xff,0x4a,0x02]
vhsub.u16 q0, q7, q5

# CHECK: vhsub.u32 q2, q6, q4  @ encoding: [0x2c,0xff,0x48,0x42]
vhsub.u32 q2, q6, q4

# CHECK: vhadd.s8 q0, q7, q0  @ encoding: [0x0e,0xef,0x40,0x00]
vhadd.s8 q0, q7, q0

# CHECK: vhadd.s16 q4, q0, q2  @ encoding: [0x10,0xef,0x44,0x80]
vhadd.s16 q4, q0, q2

# CHECK: vhadd.s32 q0, q3, q1  @ encoding: [0x26,0xef,0x42,0x00]
vhadd.s32 q0, q3, q1

# CHECK: vhadd.u8 q3, q0, q3  @ encoding: [0x00,0xff,0x46,0x60]
vhadd.u8 q3, q0, q3

# CHECK: vhadd.u16 q0, q1, q3  @ encoding: [0x12,0xff,0x46,0x00]
vhadd.u16 q0, q1, q3

# CHECK: vhadd.u32 q0, q1, q3  @ encoding: [0x22,0xff,0x46,0x00]
vhadd.u32 q0, q1, q3

# CHECK: vdup.8 q6, r8  @ encoding: [0xec,0xee,0x10,0x8b]
vdup.8 q6, r8

# CHECK: vdup.16 q7, lr  @ encoding: [0xae,0xee,0x30,0xeb]
vdup.16 q7, lr

# CHECK: vdup.32 q1, r9  @ encoding: [0xa2,0xee,0x10,0x9b]
vdup.32 q1, r9

# CHECK: vpte.i8 eq, q0, q0
# CHECK: vdupt.16 q0, r1  @ encoding: [0xa0,0xee,0x30,0x1b]
# CHECK: vdupe.16 q0, r1  @ encoding: [0xa0,0xee,0x30,0x1b]
vpte.i8 eq, q0, q0
vdupt.16 q0, r1
vdupe.16 q0, r1

# CHECK: vcls.s8 q2, q1  @ encoding: [0xb0,0xff,0x42,0x44]
vcls.s8 q2, q1

# CHECK: vcls.s16 q0, q4  @ encoding: [0xb4,0xff,0x48,0x04]
vcls.s16 q0, q4

# CHECK: vcls.s32 q0, q0  @ encoding: [0xb8,0xff,0x40,0x04]
vcls.s32 q0, q0

# CHECK: vclz.i8 q0, q7  @ encoding: [0xb0,0xff,0xce,0x04]
vclz.i8 q0, q7

# CHECK: vclz.i16 q4, q7  @ encoding: [0xb4,0xff,0xce,0x84]
vclz.i16 q4, q7

# CHECK: vclz.i32 q7, q5  @ encoding: [0xb8,0xff,0xca,0xe4]
vclz.i32 q7, q5

# CHECK: vneg.s8 q1, q0  @ encoding: [0xb1,0xff,0xc0,0x23]
vneg.s8 q1, q0

# CHECK: vneg.s16 q0, q1  @ encoding: [0xb5,0xff,0xc2,0x03]
vneg.s16 q0, q1

# CHECK: vneg.s32 q7, q2  @ encoding: [0xb9,0xff,0xc4,0xe3]
vneg.s32 q7, q2

# CHECK: vabs.s8 q1, q1  @ encoding: [0xb1,0xff,0x42,0x23]
vabs.s8 q1, q1

# CHECK: vabs.s16 q0, q2  @ encoding: [0xb5,0xff,0x44,0x03]
vabs.s16 q0, q2

# CHECK: vabs.s32 q0, q7  @ encoding: [0xb9,0xff,0x4e,0x03]
vabs.s32 q0, q7

# CHECK: vqneg.s8 q0, q0  @ encoding: [0xb0,0xff,0xc0,0x07]
vqneg.s8 q0, q0

# CHECK: vqneg.s16 q6, q2  @ encoding: [0xb4,0xff,0xc4,0xc7]
vqneg.s16 q6, q2

# CHECK: vqneg.s32 q7, q2  @ encoding: [0xb8,0xff,0xc4,0xe7]
vqneg.s32 q7, q2

# CHECK: vqabs.s8 q2, q4  @ encoding: [0xb0,0xff,0x48,0x47]
vqabs.s8 q2, q4

# CHECK: vqabs.s16 q0, q2  @ encoding: [0xb4,0xff,0x44,0x07]
vqabs.s16 q0, q2

# CHECK: vqabs.s32 q0, q5  @ encoding: [0xb8,0xff,0x4a,0x07]
vqabs.s32 q0, q5

vpste
vnegt.s8 q0, q1
vnege.s8 q0, q1
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vnegt.s8 q0, q1 @ encoding: [0xb1,0xff,0xc2,0x03]
# CHECK: vnege.s8 q0, q1 @ encoding: [0xb1,0xff,0xc2,0x03]

vpst
vqaddt.s16 q0, q1, q2
# CHECK: vpst @ encoding: [0x71,0xfe,0x4d,0x0f]
# CHECK: vqaddt.s16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x00]

vpste
vqnegt.s8 q0, q1
vqnege.s16 q0, q1
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vqnegt.s8 q0, q1 @ encoding: [0xb0,0xff,0xc2,0x07]
# CHECK: vqnege.s16 q0, q1 @ encoding: [0xb4,0xff,0xc2,0x07]

# CHECK: vmina.s8 q1, q7  @ encoding: [0x33,0xee,0x8f,0x3e]
# CHECK-NOFP: vmina.s8 q1, q7  @ encoding: [0x33,0xee,0x8f,0x3e]
vmina.s8 q1, q7

# CHECK: vmina.s16 q1, q4  @ encoding: [0x37,0xee,0x89,0x3e]
# CHECK-NOFP: vmina.s16 q1, q4  @ encoding: [0x37,0xee,0x89,0x3e]
vmina.s16 q1, q4

# CHECK: vmina.s32 q0, q7  @ encoding: [0x3b,0xee,0x8f,0x1e]
# CHECK-NOFP: vmina.s32 q0, q7  @ encoding: [0x3b,0xee,0x8f,0x1e]
vmina.s32 q0, q7

# CHECK: vmaxa.s8 q0, q7  @ encoding: [0x33,0xee,0x8f,0x0e]
# CHECK-NOFP: vmaxa.s8 q0, q7  @ encoding: [0x33,0xee,0x8f,0x0e]
vmaxa.s8 q0, q7

# CHECK: vmaxa.s16 q1, q0  @ encoding: [0x37,0xee,0x81,0x2e]
# CHECK-NOFP: vmaxa.s16 q1, q0  @ encoding: [0x37,0xee,0x81,0x2e]
vmaxa.s16 q1, q0

# CHECK: vmaxa.s32 q1, q0  @ encoding: [0x3b,0xee,0x81,0x2e]
# CHECK-NOFP: vmaxa.s32 q1, q0  @ encoding: [0x3b,0xee,0x81,0x2e]
vmaxa.s32 q1, q0
