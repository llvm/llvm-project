# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s

# CHECK: vsub.i8 q0, q3, r3  @ encoding: [0x07,0xee,0x43,0x1f]
# CHECK-NOFP: vsub.i8 q0, q3, r3  @ encoding: [0x07,0xee,0x43,0x1f]
vsub.i8 q0, q3, r3

# CHECK: vsub.i16 q0, q7, lr  @ encoding: [0x1f,0xee,0x4e,0x1f]
# CHECK-NOFP: vsub.i16 q0, q7, lr  @ encoding: [0x1f,0xee,0x4e,0x1f]
vsub.i16 q0, q7, lr

# CHECK: vsub.i32 q1, q5, r10  @ encoding: [0x2b,0xee,0x4a,0x3f]
# CHECK-NOFP: vsub.i32 q1, q5, r10  @ encoding: [0x2b,0xee,0x4a,0x3f]
vsub.i32 q1, q5, r10

# CHECK: vadd.i8 q1, q4, r7  @ encoding: [0x09,0xee,0x47,0x2f]
# CHECK-NOFP: vadd.i8 q1, q4, r7  @ encoding: [0x09,0xee,0x47,0x2f]
vadd.i8 q1, q4, r7

# CHECK: vadd.i16 q0, q6, r11  @ encoding: [0x1d,0xee,0x4b,0x0f]
# CHECK-NOFP: vadd.i16 q0, q6, r11  @ encoding: [0x1d,0xee,0x4b,0x0f]
vadd.i16 q0, q6, r11

# CHECK: vadd.i32 q0, q1, r6  @ encoding: [0x23,0xee,0x46,0x0f]
# CHECK-NOFP: vadd.i32 q0, q1, r6  @ encoding: [0x23,0xee,0x46,0x0f]
vadd.i32 q0, q1, r6

# CHECK: vqsub.s8 q2, q2, r8  @ encoding: [0x04,0xee,0x68,0x5f]
# CHECK-NOFP: vqsub.s8 q2, q2, r8  @ encoding: [0x04,0xee,0x68,0x5f]
vqsub.s8 q2, q2, r8

# CHECK: vqsub.s16 q1, q4, r0  @ encoding: [0x18,0xee,0x60,0x3f]
# CHECK-NOFP: vqsub.s16 q1, q4, r0  @ encoding: [0x18,0xee,0x60,0x3f]
vqsub.s16 q1, q4, r0

# CHECK: vqsub.s32 q0, q2, r0  @ encoding: [0x24,0xee,0x60,0x1f]
# CHECK-NOFP: vqsub.s32 q0, q2, r0  @ encoding: [0x24,0xee,0x60,0x1f]
vqsub.s32 q0, q2, r0

# CHECK: vqsub.u8 q0, q1, r2  @ encoding: [0x02,0xfe,0x62,0x1f]
# CHECK-NOFP: vqsub.u8 q0, q1, r2  @ encoding: [0x02,0xfe,0x62,0x1f]
vqsub.u8 q0, q1, r2

# CHECK: vqsub.u16 q0, q2, r6  @ encoding: [0x14,0xfe,0x66,0x1f]
# CHECK-NOFP: vqsub.u16 q0, q2, r6  @ encoding: [0x14,0xfe,0x66,0x1f]
vqsub.u16 q0, q2, r6

# CHECK: vqsub.u32 q0, q2, r2  @ encoding: [0x24,0xfe,0x62,0x1f]
# CHECK-NOFP: vqsub.u32 q0, q2, r2  @ encoding: [0x24,0xfe,0x62,0x1f]
vqsub.u32 q0, q2, r2

# CHECK: vqadd.s8 q0, q6, r1  @ encoding: [0x0c,0xee,0x61,0x0f]
# CHECK-NOFP: vqadd.s8 q0, q6, r1  @ encoding: [0x0c,0xee,0x61,0x0f]
vqadd.s8 q0, q6, r1

# CHECK: vqadd.s16 q3, q4, r2  @ encoding: [0x18,0xee,0x62,0x6f]
# CHECK-NOFP: vqadd.s16 q3, q4, r2  @ encoding: [0x18,0xee,0x62,0x6f]
vqadd.s16 q3, q4, r2

# CHECK: vqadd.s32 q0, q5, r11  @ encoding: [0x2a,0xee,0x6b,0x0f]
# CHECK-NOFP: vqadd.s32 q0, q5, r11  @ encoding: [0x2a,0xee,0x6b,0x0f]
vqadd.s32 q0, q5, r11

# CHECK: vqadd.u8 q0, q1, r8  @ encoding: [0x02,0xfe,0x68,0x0f]
# CHECK-NOFP: vqadd.u8 q0, q1, r8  @ encoding: [0x02,0xfe,0x68,0x0f]
vqadd.u8 q0, q1, r8

# CHECK: vqadd.u16 q0, q5, r9  @ encoding: [0x1a,0xfe,0x69,0x0f]
# CHECK-NOFP: vqadd.u16 q0, q5, r9  @ encoding: [0x1a,0xfe,0x69,0x0f]
vqadd.u16 q0, q5, r9

# CHECK: vqadd.u32 q0, q0, r7  @ encoding: [0x20,0xfe,0x67,0x0f]
# CHECK-NOFP: vqadd.u32 q0, q0, r7  @ encoding: [0x20,0xfe,0x67,0x0f]
vqadd.u32 q0, q0, r7

# CHECK: vqdmullb.s16 q0, q1, r6  @ encoding: [0x32,0xee,0x66,0x0f]
# CHECK-NOFP: vqdmullb.s16 q0, q1, r6  @ encoding: [0x32,0xee,0x66,0x0f]
vqdmullb.s16 q0, q1, r6

# CHECK: vqdmullb.s32 q0, q3, q7  @ encoding: [0x36,0xfe,0x0f,0x0f]
# CHECK-NOFP: vqdmullb.s32 q0, q3, q7  @ encoding: [0x36,0xfe,0x0f,0x0f]
vqdmullb.s32 q0, q3, q7

# CHECK: vqdmullt.s16 q0, q1, r0  @ encoding: [0x32,0xee,0x60,0x1f]
# CHECK-NOFP: vqdmullt.s16 q0, q1, r0  @ encoding: [0x32,0xee,0x60,0x1f]
vqdmullt.s16 q0, q1, r0

# CHECK: vqdmullt.s32 q0, q4, r5  @ encoding: [0x38,0xfe,0x65,0x1f]
# CHECK-NOFP: vqdmullt.s32 q0, q4, r5  @ encoding: [0x38,0xfe,0x65,0x1f]
vqdmullt.s32 q0, q4, r5

# CHECK: vsub.f16 q0, q3, r7  @ encoding: [0x36,0xfe,0x47,0x1f]
# CHECK-NOFP-NOT: vsub.f16 q0, q3, r7  @ encoding: [0x36,0xfe,0x47,0x1f]
vsub.f16 q0, q3, r7

# CHECK: vsub.f32 q1, q1, r10  @ encoding: [0x32,0xee,0x4a,0x3f]
# CHECK-NOFP-NOT: vsub.f32 q1, q1, r10  @ encoding: [0x32,0xee,0x4a,0x3f]
vsub.f32 q1, q1, r10

# CHECK: vadd.f16 q0, q1, lr  @ encoding: [0x32,0xfe,0x4e,0x0f]
# CHECK-NOFP-NOT: vadd.f16 q0, q1, lr  @ encoding: [0x32,0xfe,0x4e,0x0f]
vadd.f16 q0, q1, lr

# CHECK: vadd.f32 q1, q4, r4  @ encoding: [0x38,0xee,0x44,0x2f]
# CHECK-NOFP-NOT: vadd.f32 q1, q4, r4  @ encoding: [0x38,0xee,0x44,0x2f]
vadd.f32 q1, q4, r4

# CHECK: vhsub.s8 q0, q3, lr  @ encoding: [0x06,0xee,0x4e,0x1f]
# CHECK-NOFP: vhsub.s8 q0, q3, lr  @ encoding: [0x06,0xee,0x4e,0x1f]
vhsub.s8 q0, q3, lr

# CHECK: vhsub.s16 q0, q0, r6  @ encoding: [0x10,0xee,0x46,0x1f]
# CHECK-NOFP: vhsub.s16 q0, q0, r6  @ encoding: [0x10,0xee,0x46,0x1f]
vhsub.s16 q0, q0, r6

# CHECK: vhsub.s32 q1, q2, r7  @ encoding: [0x24,0xee,0x47,0x3f]
# CHECK-NOFP: vhsub.s32 q1, q2, r7  @ encoding: [0x24,0xee,0x47,0x3f]
vhsub.s32 q1, q2, r7

# CHECK: vhsub.u8 q1, q6, r5  @ encoding: [0x0c,0xfe,0x45,0x3f]
# CHECK-NOFP: vhsub.u8 q1, q6, r5  @ encoding: [0x0c,0xfe,0x45,0x3f]
vhsub.u8 q1, q6, r5

# CHECK: vhsub.u16 q0, q4, r10  @ encoding: [0x18,0xfe,0x4a,0x1f]
# CHECK-NOFP: vhsub.u16 q0, q4, r10  @ encoding: [0x18,0xfe,0x4a,0x1f]
vhsub.u16 q0, q4, r10

# CHECK: vhsub.u32 q0, q4, r12  @ encoding: [0x28,0xfe,0x4c,0x1f]
# CHECK-NOFP: vhsub.u32 q0, q4, r12  @ encoding: [0x28,0xfe,0x4c,0x1f]
vhsub.u32 q0, q4, r12

# CHECK: vhadd.s8 q0, q2, r1  @ encoding: [0x04,0xee,0x41,0x0f]
# CHECK-NOFP: vhadd.s8 q0, q2, r1  @ encoding: [0x04,0xee,0x41,0x0f]
vhadd.s8 q0, q2, r1

# CHECK: vhadd.s16 q0, q2, r1  @ encoding: [0x14,0xee,0x41,0x0f]
# CHECK-NOFP: vhadd.s16 q0, q2, r1  @ encoding: [0x14,0xee,0x41,0x0f]
vhadd.s16 q0, q2, r1

# CHECK: vhadd.s32 q0, q0, r10  @ encoding: [0x20,0xee,0x4a,0x0f]
# CHECK-NOFP: vhadd.s32 q0, q0, r10  @ encoding: [0x20,0xee,0x4a,0x0f]
vhadd.s32 q0, q0, r10

# CHECK: vhadd.u8 q0, q5, lr  @ encoding: [0x0a,0xfe,0x4e,0x0f]
# CHECK-NOFP: vhadd.u8 q0, q5, lr  @ encoding: [0x0a,0xfe,0x4e,0x0f]
vhadd.u8 q0, q5, lr

# CHECK: vhadd.u16 q1, q2, r2  @ encoding: [0x14,0xfe,0x42,0x2f]
# CHECK-NOFP: vhadd.u16 q1, q2, r2  @ encoding: [0x14,0xfe,0x42,0x2f]
vhadd.u16 q1, q2, r2

# CHECK: vhadd.u32 q0, q2, r11  @ encoding: [0x24,0xfe,0x4b,0x0f]
# CHECK-NOFP: vhadd.u32 q0, q2, r11  @ encoding: [0x24,0xfe,0x4b,0x0f]
vhadd.u32 q0, q2, r11

# CHECK: vqrshl.s8 q0, r0  @ encoding: [0x33,0xee,0xe0,0x1e]
# CHECK-NOFP: vqrshl.s8 q0, r0  @ encoding: [0x33,0xee,0xe0,0x1e]
vqrshl.s8 q0, r0

# CHECK: vqrshl.s16 q0, r3  @ encoding: [0x37,0xee,0xe3,0x1e]
# CHECK-NOFP: vqrshl.s16 q0, r3  @ encoding: [0x37,0xee,0xe3,0x1e]
vqrshl.s16 q0, r3

# CHECK: vqrshl.s32 q0, lr  @ encoding: [0x3b,0xee,0xee,0x1e]
# CHECK-NOFP: vqrshl.s32 q0, lr  @ encoding: [0x3b,0xee,0xee,0x1e]
vqrshl.s32 q0, lr

# CHECK: vqrshl.u8 q0, r0  @ encoding: [0x33,0xfe,0xe0,0x1e]
# CHECK-NOFP: vqrshl.u8 q0, r0  @ encoding: [0x33,0xfe,0xe0,0x1e]
vqrshl.u8 q0, r0

# CHECK: vqrshl.u16 q0, r2  @ encoding: [0x37,0xfe,0xe2,0x1e]
# CHECK-NOFP: vqrshl.u16 q0, r2  @ encoding: [0x37,0xfe,0xe2,0x1e]
vqrshl.u16 q0, r2

# CHECK: vqrshl.u32 q0, r3  @ encoding: [0x3b,0xfe,0xe3,0x1e]
# CHECK-NOFP: vqrshl.u32 q0, r3  @ encoding: [0x3b,0xfe,0xe3,0x1e]
vqrshl.u32 q0, r3

# CHECK: vqshl.s8 q0, r0  @ encoding: [0x31,0xee,0xe0,0x1e]
# CHECK-NOFP: vqshl.s8 q0, r0  @ encoding: [0x31,0xee,0xe0,0x1e]
vqshl.s8 q0, r0

# CHECK: vqshl.s16 q1, r1  @ encoding: [0x35,0xee,0xe1,0x3e]
# CHECK-NOFP: vqshl.s16 q1, r1  @ encoding: [0x35,0xee,0xe1,0x3e]
vqshl.s16 q1, r1

# CHECK: vqshl.s32 q0, r3  @ encoding: [0x39,0xee,0xe3,0x1e]
# CHECK-NOFP: vqshl.s32 q0, r3  @ encoding: [0x39,0xee,0xe3,0x1e]
vqshl.s32 q0, r3

# CHECK: vqshl.u8 q0, r1  @ encoding: [0x31,0xfe,0xe1,0x1e]
# CHECK-NOFP: vqshl.u8 q0, r1  @ encoding: [0x31,0xfe,0xe1,0x1e]
vqshl.u8 q0, r1

# CHECK: vqshl.u16 q0, r11  @ encoding: [0x35,0xfe,0xeb,0x1e]
# CHECK-NOFP: vqshl.u16 q0, r11  @ encoding: [0x35,0xfe,0xeb,0x1e]
vqshl.u16 q0, r11

# CHECK: vqshl.u32 q0, lr  @ encoding: [0x39,0xfe,0xee,0x1e]
# CHECK-NOFP: vqshl.u32 q0, lr  @ encoding: [0x39,0xfe,0xee,0x1e]
vqshl.u32 q0, lr

# CHECK: vrshl.s8 q0, r6  @ encoding: [0x33,0xee,0x66,0x1e]
# CHECK-NOFP: vrshl.s8 q0, r6  @ encoding: [0x33,0xee,0x66,0x1e]
vrshl.s8 q0, r6

# CHECK: vrshl.s16 q0, lr  @ encoding: [0x37,0xee,0x6e,0x1e]
# CHECK-NOFP: vrshl.s16 q0, lr  @ encoding: [0x37,0xee,0x6e,0x1e]
vrshl.s16 q0, lr

# CHECK: vrshl.s32 q0, r4  @ encoding: [0x3b,0xee,0x64,0x1e]
# CHECK-NOFP: vrshl.s32 q0, r4  @ encoding: [0x3b,0xee,0x64,0x1e]
vrshl.s32 q0, r4

# CHECK: vrshl.u8 q0, r0  @ encoding: [0x33,0xfe,0x60,0x1e]
# CHECK-NOFP: vrshl.u8 q0, r0  @ encoding: [0x33,0xfe,0x60,0x1e]
vrshl.u8 q0, r0

# CHECK: vrshl.u16 q0, r10  @ encoding: [0x37,0xfe,0x6a,0x1e]
# CHECK-NOFP: vrshl.u16 q0, r10  @ encoding: [0x37,0xfe,0x6a,0x1e]
vrshl.u16 q0, r10

# CHECK: vrshl.u32 q0, r1  @ encoding: [0x3b,0xfe,0x61,0x1e]
# CHECK-NOFP: vrshl.u32 q0, r1  @ encoding: [0x3b,0xfe,0x61,0x1e]
vrshl.u32 q0, r1

# CHECK: vshl.s8 q0, lr  @ encoding: [0x31,0xee,0x6e,0x1e]
# CHECK-NOFP: vshl.s8 q0, lr  @ encoding: [0x31,0xee,0x6e,0x1e]
vshl.s8 q0, lr

# CHECK: vshl.s16 q0, lr  @ encoding: [0x35,0xee,0x6e,0x1e]
# CHECK-NOFP: vshl.s16 q0, lr  @ encoding: [0x35,0xee,0x6e,0x1e]
vshl.s16 q0, lr

# CHECK: vshl.s32 q0, r1  @ encoding: [0x39,0xee,0x61,0x1e]
# CHECK-NOFP: vshl.s32 q0, r1  @ encoding: [0x39,0xee,0x61,0x1e]
vshl.s32 q0, r1

# CHECK: vshl.u8 q0, r10  @ encoding: [0x31,0xfe,0x6a,0x1e]
# CHECK-NOFP: vshl.u8 q0, r10  @ encoding: [0x31,0xfe,0x6a,0x1e]
vshl.u8 q0, r10

# CHECK: vshl.u16 q1, r10  @ encoding: [0x35,0xfe,0x6a,0x3e]
# CHECK-NOFP: vshl.u16 q1, r10  @ encoding: [0x35,0xfe,0x6a,0x3e]
vshl.u16 q1, r10

# CHECK: vshl.u32 q0, r12  @ encoding: [0x39,0xfe,0x6c,0x1e]
# CHECK-NOFP: vshl.u32 q0, r12  @ encoding: [0x39,0xfe,0x6c,0x1e]
vshl.u32 q0, r12

# CHECK: vbrsr.8 q0, q4, r8  @ encoding: [0x09,0xfe,0x68,0x1e]
# CHECK-NOFP: vbrsr.8 q0, q4, r8  @ encoding: [0x09,0xfe,0x68,0x1e]
vbrsr.8 q0, q4, r8

# CHECK: vbrsr.16 q0, q1, r1  @ encoding: [0x13,0xfe,0x61,0x1e]
# CHECK-NOFP: vbrsr.16 q0, q1, r1  @ encoding: [0x13,0xfe,0x61,0x1e]
vbrsr.16 q0, q1, r1

# CHECK: vbrsr.32 q0, q6, r0  @ encoding: [0x2d,0xfe,0x60,0x1e]
# CHECK-NOFP: vbrsr.32 q0, q6, r0  @ encoding: [0x2d,0xfe,0x60,0x1e]
vbrsr.32 q0, q6, r0

# CHECK: vmul.i8 q0, q0, r12  @ encoding: [0x01,0xee,0x6c,0x1e]
# CHECK-NOFP: vmul.i8 q0, q0, r12  @ encoding: [0x01,0xee,0x6c,0x1e]
vmul.i8 q0, q0, r12

# CHECK: vmul.i16 q0, q4, r7  @ encoding: [0x19,0xee,0x67,0x1e]
# CHECK-NOFP: vmul.i16 q0, q4, r7  @ encoding: [0x19,0xee,0x67,0x1e]
vmul.i16 q0, q4, r7

# CHECK: vmul.i32 q0, q1, r11  @ encoding: [0x23,0xee,0x6b,0x1e]
# CHECK-NOFP: vmul.i32 q0, q1, r11  @ encoding: [0x23,0xee,0x6b,0x1e]
vmul.i32 q0, q1, r11

# CHECK: vmul.f16 q0, q0, r10  @ encoding: [0x31,0xfe,0x6a,0x0e]
# CHECK-NOFP-NOT: vmul.f16 q0, q0, r10  @ encoding: [0x31,0xfe,0x6a,0x0e]
vmul.f16 q0, q0, r10

# CHECK: vmul.f32 q0, q1, r7  @ encoding: [0x33,0xee,0x67,0x0e]
# CHECK-NOFP-NOT: vmul.f32 q0, q1, r7  @ encoding: [0x33,0xee,0x67,0x0e]
vmul.f32 q0, q1, r7

# CHECK: vqdmulh.s8 q0, q1, r6  @ encoding: [0x03,0xee,0x66,0x0e]
# CHECK-NOFP: vqdmulh.s8 q0, q1, r6  @ encoding: [0x03,0xee,0x66,0x0e]
vqdmulh.s8 q0, q1, r6

# CHECK: vqdmulh.s16 q0, q2, r2  @ encoding: [0x15,0xee,0x62,0x0e]
# CHECK-NOFP: vqdmulh.s16 q0, q2, r2  @ encoding: [0x15,0xee,0x62,0x0e]
vqdmulh.s16 q0, q2, r2

# CHECK: vqdmulh.s32 q1, q3, r8  @ encoding: [0x27,0xee,0x68,0x2e]
# CHECK-NOFP: vqdmulh.s32 q1, q3, r8  @ encoding: [0x27,0xee,0x68,0x2e]
vqdmulh.s32 q1, q3, r8

# CHECK: vqrdmulh.s8 q0, q2, r6  @ encoding: [0x05,0xfe,0x66,0x0e]
# CHECK-NOFP: vqrdmulh.s8 q0, q2, r6  @ encoding: [0x05,0xfe,0x66,0x0e]
vqrdmulh.s8 q0, q2, r6

# CHECK: vqrdmulh.s16 q0, q0, r2  @ encoding: [0x11,0xfe,0x62,0x0e]
# CHECK-NOFP: vqrdmulh.s16 q0, q0, r2  @ encoding: [0x11,0xfe,0x62,0x0e]
vqrdmulh.s16 q0, q0, r2

# CHECK: vqrdmulh.s32 q0, q0, r2  @ encoding: [0x21,0xfe,0x62,0x0e]
# CHECK-NOFP: vqrdmulh.s32 q0, q0, r2  @ encoding: [0x21,0xfe,0x62,0x0e]
vqrdmulh.s32 q0, q0, r2

# CHECK: vfmas.f16 q0, q0, r12  @ encoding: [0x31,0xfe,0x4c,0x1e]
# CHECK-NOFP-NOT: vfmas.f16 q0, q0, r12  @ encoding: [0x31,0xfe,0x4c,0x1e]
vfmas.f16 q0, q0, r12

# CHECK: vfmas.f32 q0, q3, lr  @ encoding: [0x37,0xee,0x4e,0x1e]
# CHECK-NOFP-NOT: vfmas.f32 q0, q3, lr  @ encoding: [0x37,0xee,0x4e,0x1e]
vfmas.f32 q0, q3, lr

# CHECK: vmlas.i8 q0, q0, r6  @ encoding: [0x01,0xee,0x46,0x1e]
# CHECK-NOFP: vmlas.i8 q0, q0, r6  @ encoding: [0x01,0xee,0x46,0x1e]
vmlas.i8 q0, q0, r6

# CHECK: vmlas.i16 q0, q2, r9  @ encoding: [0x15,0xee,0x49,0x1e]
# CHECK-NOFP: vmlas.i16 q0, q2, r9  @ encoding: [0x15,0xee,0x49,0x1e]
vmlas.i16 q0, q2, r9

# CHECK: vmlas.i32 q0, q7, r6  @ encoding: [0x2f,0xee,0x46,0x1e]
# CHECK-NOFP: vmlas.i32 q0, q7, r6  @ encoding: [0x2f,0xee,0x46,0x1e]
vmlas.i32 q0, q7, r6

# CHECK: vmlas.i8 q0, q0, r6  @ encoding: [0x01,0xee,0x46,0x1e]
# CHECK-NOFP: vmlas.i8 q0, q0, r6  @ encoding: [0x01,0xee,0x46,0x1e]
vmlas.s8 q0, q0, r6

# CHECK: vmlas.i16 q0, q2, r9  @ encoding: [0x15,0xee,0x49,0x1e]
# CHECK-NOFP: vmlas.i16 q0, q2, r9  @ encoding: [0x15,0xee,0x49,0x1e]
vmlas.s16 q0, q2, r9

# CHECK: vmlas.i32 q0, q7, r6  @ encoding: [0x2f,0xee,0x46,0x1e]
# CHECK-NOFP: vmlas.i32 q0, q7, r6  @ encoding: [0x2f,0xee,0x46,0x1e]
vmlas.s32 q0, q7, r6

# CHECK: vmlas.i8 q0, q5, lr  @ encoding: [0x0b,0xee,0x4e,0x1e]
# CHECK-NOFP: vmlas.i8 q0, q5, lr  @ encoding: [0x0b,0xee,0x4e,0x1e]
vmlas.u8 q0, q5, lr

# CHECK: vmlas.i16 q0, q3, r12  @ encoding: [0x17,0xee,0x4c,0x1e]
# CHECK-NOFP: vmlas.i16 q0, q3, r12  @ encoding: [0x17,0xee,0x4c,0x1e]
vmlas.u16 q0, q3, r12

# CHECK: vmlas.i32 q1, q1, r11  @ encoding: [0x23,0xee,0x4b,0x3e]
# CHECK-NOFP: vmlas.i32 q1, q1, r11  @ encoding: [0x23,0xee,0x4b,0x3e]
vmlas.u32 q1, q1, r11

# CHECK: vfma.f16 q1, q1, r6  @ encoding: [0x33,0xfe,0x46,0x2e]
# CHECK-NOFP-NOT: vfma.f16 q1, q1, r6  @ encoding: [0x33,0xfe,0x46,0x2e]
vfma.f16 q1, q1, r6

# CHECK: vfmas.f32 q7, q4, r6  @ encoding: [0x39,0xee,0x46,0xfe]
# CHECK-NOFP-NOT: vfmas.f32 q7, q4, r6  @ encoding: [0x39,0xee,0x46,0xfe]
vfmas.f32 q7, q4, r6

# CHECK: vmla.i8 q0, q3, r8  @ encoding: [0x07,0xee,0x48,0x0e]
# CHECK-NOFP: vmla.i8 q0, q3, r8  @ encoding: [0x07,0xee,0x48,0x0e]
vmla.i8 q0, q3, r8

# CHECK: vmla.i16 q1, q3, r10  @ encoding: [0x17,0xee,0x4a,0x2e]
# CHECK-NOFP: vmla.i16 q1, q3, r10  @ encoding: [0x17,0xee,0x4a,0x2e]
vmla.i16 q1, q3, r10

# CHECK: vmla.i32 q1, q3, r1  @ encoding: [0x27,0xee,0x41,0x2e]
# CHECK-NOFP: vmla.i32 q1, q3, r1  @ encoding: [0x27,0xee,0x41,0x2e]
vmla.i32 q1, q3, r1

# CHECK: vmla.i8 q0, q3, r8  @ encoding: [0x07,0xee,0x48,0x0e]
# CHECK-NOFP: vmla.i8 q0, q3, r8  @ encoding: [0x07,0xee,0x48,0x0e]
vmla.s8 q0, q3, r8

# CHECK: vmla.i16 q1, q3, r10  @ encoding: [0x17,0xee,0x4a,0x2e]
# CHECK-NOFP: vmla.i16 q1, q3, r10  @ encoding: [0x17,0xee,0x4a,0x2e]
vmla.s16 q1, q3, r10

# CHECK: vmla.i32 q1, q3, r1  @ encoding: [0x27,0xee,0x41,0x2e]
# CHECK-NOFP: vmla.i32 q1, q3, r1  @ encoding: [0x27,0xee,0x41,0x2e]
vmla.s32 q1, q3, r1

# CHECK: vmla.i8 q0, q7, r10  @ encoding: [0x0f,0xee,0x4a,0x0e]
# CHECK-NOFP: vmla.i8 q0, q7, r10  @ encoding: [0x0f,0xee,0x4a,0x0e]
vmla.u8 q0, q7, r10

# CHECK: vmla.i16 q0, q0, r7  @ encoding: [0x11,0xee,0x47,0x0e]
# CHECK-NOFP: vmla.i16 q0, q0, r7  @ encoding: [0x11,0xee,0x47,0x0e]
vmla.u16 q0, q0, r7

# CHECK: vmla.i32 q1, q6, r10  @ encoding: [0x2d,0xee,0x4a,0x2e]
# CHECK-NOFP: vmla.i32 q1, q6, r10  @ encoding: [0x2d,0xee,0x4a,0x2e]
vmla.u32 q1, q6, r10

# CHECK: vqdmlash.s8 q0, q0, r5  @ encoding: [0x00,0xee,0x65,0x1e]
# CHECK-NOFP: vqdmlash.s8 q0, q0, r5  @ encoding: [0x00,0xee,0x65,0x1e]
vqdmlash.s8 q0, q0, r5

# CHECK: vqdmlash.s16 q0, q5, lr  @ encoding: [0x1a,0xee,0x6e,0x1e]
# CHECK-NOFP: vqdmlash.s16 q0, q5, lr  @ encoding: [0x1a,0xee,0x6e,0x1e]
vqdmlash.s16 q0, q5, lr

# CHECK: vqdmlash.s32 q0, q2, r3  @ encoding: [0x24,0xee,0x63,0x1e]
# CHECK-NOFP: vqdmlash.s32 q0, q2, r3  @ encoding: [0x24,0xee,0x63,0x1e]
vqdmlash.s32 q0, q2, r3

# ERROR: [[@LINE+2]]:9: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:9: error: invalid operand for instruction
vqdmlash.u8 q0, q4, r2

# ERROR: [[@LINE+2]]:9: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:9: error: invalid operand for instruction
vqdmlash.u16 q1, q4, r2

# ERROR: [[@LINE+2]]:9: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:9: error: invalid operand for instruction
vqdmlash.u32 q1, q5, r0

# CHECK: vqdmlah.s8 q0, q3, r3  @ encoding: [0x06,0xee,0x63,0x0e]
# CHECK-NOFP: vqdmlah.s8 q0, q3, r3  @ encoding: [0x06,0xee,0x63,0x0e]
vqdmlah.s8 q0, q3, r3

# CHECK: vqdmlah.s16 q5, q3, r9  @ encoding: [0x16,0xee,0x69,0xae]
# CHECK-NOFP: vqdmlah.s16 q5, q3, r9  @ encoding: [0x16,0xee,0x69,0xae]
vqdmlah.s16 q5, q3, r9

# CHECK: vqdmlah.s32 q0, q1, r11  @ encoding: [0x22,0xee,0x6b,0x0e]
# CHECK-NOFP: vqdmlah.s32 q0, q1, r11  @ encoding: [0x22,0xee,0x6b,0x0e]
vqdmlah.s32 q0, q1, r11

# ERROR: [[@LINE+2]]:8: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:8: error: invalid operand for instruction
vqdmlah.u8 q0, q2, lr

# ERROR: [[@LINE+2]]:8: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:8: error: invalid operand for instruction
vqdmlah.u16 q0, q3, r10

# ERROR: [[@LINE+2]]:8: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:8: error: invalid operand for instruction
vqdmlah.u32 q1, q5, r2

# CHECK: vqrdmlash.s8 q0, q5, r10  @ encoding: [0x0a,0xee,0x4a,0x1e]
# CHECK-NOFP: vqrdmlash.s8 q0, q5, r10  @ encoding: [0x0a,0xee,0x4a,0x1e]
vqrdmlash.s8 q0, q5, r10

# CHECK: vqrdmlash.s16 q0, q3, r2  @ encoding: [0x16,0xee,0x42,0x1e]
# CHECK-NOFP: vqrdmlash.s16 q0, q3, r2  @ encoding: [0x16,0xee,0x42,0x1e]
vqrdmlash.s16 q0, q3, r2

# CHECK: vqrdmlash.s32 q0, q0, r4  @ encoding: [0x20,0xee,0x44,0x1e]
# CHECK-NOFP: vqrdmlash.s32 q0, q0, r4  @ encoding: [0x20,0xee,0x44,0x1e]
vqrdmlash.s32 q0, q0, r4

# ERROR: [[@LINE+2]]:10: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:10: error: invalid operand for instruction
vqrdmlash.u8 q0, q4, r9

# ERROR: [[@LINE+2]]:10: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:10: error: invalid operand for instruction
vqrdmlash.u16 q0, q6, r12

# ERROR: [[@LINE+2]]:10: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:10: error: invalid operand for instruction
vqrdmlash.u32 q0, q3, r7

# CHECK: vqrdmlah.s8 q0, q5, r11  @ encoding: [0x0a,0xee,0x4b,0x0e]
# CHECK-NOFP: vqrdmlah.s8 q0, q5, r11  @ encoding: [0x0a,0xee,0x4b,0x0e]
vqrdmlah.s8 q0, q5, r11

# CHECK: vqrdmlah.s16 q0, q2, r10  @ encoding: [0x14,0xee,0x4a,0x0e]
# CHECK-NOFP: vqrdmlah.s16 q0, q2, r10  @ encoding: [0x14,0xee,0x4a,0x0e]
vqrdmlah.s16 q0, q2, r10

# CHECK: vqrdmlah.s32 q0, q4, r11  @ encoding: [0x28,0xee,0x4b,0x0e]
# CHECK-NOFP: vqrdmlah.s32 q0, q4, r11  @ encoding: [0x28,0xee,0x4b,0x0e]
vqrdmlah.s32 q0, q4, r11

# ERROR: [[@LINE+2]]:9: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:9: error: invalid operand for instruction
vqrdmlah.u8 q0, q4, r2

# ERROR: [[@LINE+2]]:9: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:9: error: invalid operand for instruction
vqrdmlah.u16 q0, q6, r1

# ERROR: [[@LINE+2]]:9: error: invalid operand for instruction
# ERROR-NOFP: [[@LINE+1]]:9: error: invalid operand for instruction
vqrdmlah.u32 q0, q4, r2

# CHECK: viwdup.u8 q0, lr, r1, #1  @ encoding: [0x0f,0xee,0x60,0x0f]
# CHECK-NOFP: viwdup.u8 q0, lr, r1, #1  @ encoding: [0x0f,0xee,0x60,0x0f]
viwdup.u8 q0, lr, r1, #1

# CHECK: viwdup.u16 q1, r10, r1, #8  @ encoding: [0x1b,0xee,0xe1,0x2f]
# CHECK-NOFP: viwdup.u16 q1, r10, r1, #8  @ encoding: [0x1b,0xee,0xe1,0x2f]
viwdup.u16 q1, r10, r1, #8

# CHECK: viwdup.u32 q6, r10, r5, #4  @ encoding: [0x2b,0xee,0xe4,0xcf]
# CHECK-NOFP: viwdup.u32 q6, r10, r5, #4  @ encoding: [0x2b,0xee,0xe4,0xcf]
viwdup.u32 q6, r10, r5, #4

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: vector increment immediate must be 1, 2, 4 or 8
viwdup.u32 q6, r10, r5, #3

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an even-numbered register
viwdup.u32 q6, r3, r5, #4

# CHECK: vdwdup.u8 q0, r12, r11, #8  @ encoding: [0x0d,0xee,0xeb,0x1f]
# CHECK-NOFP: vdwdup.u8 q0, r12, r11, #8  @ encoding: [0x0d,0xee,0xeb,0x1f]
vdwdup.u8 q0, r12, r11, #8

# CHECK: vdwdup.u16 q0, r12, r1, #2  @ encoding: [0x1d,0xee,0x61,0x1f]
# CHECK-NOFP: vdwdup.u16 q0, r12, r1, #2  @ encoding: [0x1d,0xee,0x61,0x1f]
vdwdup.u16 q0, r12, r1, #2

# CHECK: vdwdup.u32 q0, r0, r7, #8  @ encoding: [0x21,0xee,0xe7,0x1f]
# CHECK-NOFP: vdwdup.u32 q0, r0, r7, #8  @ encoding: [0x21,0xee,0xe7,0x1f]
vdwdup.u32 q0, r0, r7, #8

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: vector increment immediate must be 1, 2, 4 or 8
vdwdup.u32 q0, r0, r7, #9

# CHECK: vidup.u8 q0, lr, #2  @ encoding: [0x0f,0xee,0x6f,0x0f]
# CHECK-NOFP: vidup.u8 q0, lr, #2  @ encoding: [0x0f,0xee,0x6f,0x0f]
vidup.u8 q0, lr, #2

# CHECK: vidup.u16 q0, lr, #4  @ encoding: [0x1f,0xee,0xee,0x0f]
# CHECK-NOFP: vidup.u16 q0, lr, #4  @ encoding: [0x1f,0xee,0xee,0x0f]
vidup.u16 q0, lr, #4

# CHECK: vidup.u32 q0, r12, #1  @ encoding: [0x2d,0xee,0x6e,0x0f]
# CHECK-NOFP: vidup.u32 q0, r12, #1  @ encoding: [0x2d,0xee,0x6e,0x0f]
vidup.u32 q0, r12, #1

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: vector increment immediate must be 1, 2, 4 or 8
vidup.u32 q0, r12, #3

# CHECK: vddup.u8 q0, r4, #4  @ encoding: [0x05,0xee,0xee,0x1f]
# CHECK-NOFP: vddup.u8 q0, r4, #4  @ encoding: [0x05,0xee,0xee,0x1f]
vddup.u8 q0, r4, #4

# CHECK: vddup.u16 q0, r10, #4  @ encoding: [0x1b,0xee,0xee,0x1f]
# CHECK-NOFP: vddup.u16 q0, r10, #4  @ encoding: [0x1b,0xee,0xee,0x1f]
vddup.u16 q0, r10, #4

# CHECK: vddup.u32 q2, r0, #8  @ encoding: [0x21,0xee,0xef,0x5f]
# CHECK-NOFP: vddup.u32 q2, r0, #8  @ encoding: [0x21,0xee,0xef,0x5f]
vddup.u32 q2, r0, #8

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: vector increment immediate must be 1, 2, 4 or 8
vddup.u32 q2, r0, #5

# CHECK: vctp.8 lr  @ encoding: [0x0e,0xf0,0x01,0xe8]
# CHECK-NOFP: vctp.8 lr  @ encoding: [0x0e,0xf0,0x01,0xe8]
vctp.8 lr

# CHECK: vctp.16 r0  @ encoding: [0x10,0xf0,0x01,0xe8]
# CHECK-NOFP: vctp.16 r0  @ encoding: [0x10,0xf0,0x01,0xe8]
vctp.16 r0

# CHECK: vctp.32 r10  @ encoding: [0x2a,0xf0,0x01,0xe8]
# CHECK-NOFP: vctp.32 r10  @ encoding: [0x2a,0xf0,0x01,0xe8]
vctp.32 r10

# CHECK: vctp.64 r1  @ encoding: [0x31,0xf0,0x01,0xe8]
# CHECK-NOFP: vctp.64 r1  @ encoding: [0x31,0xf0,0x01,0xe8]
vctp.64 r1

vpste
vmult.i8 q0, q1, q2
vmule.i16 q0, q1, q2
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vmult.i8  q0, q1, q2 @ encoding: [0x02,0xef,0x54,0x09]
# CHECK-NOFP: vmult.i8  q0, q1, q2 @ encoding: [0x02,0xef,0x54,0x09]
# CHECK: vmule.i16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x09]
# CHECK-NOFP: vmule.i16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x09]

vpste
vmult.i16 q0, q1, q2
vmule.i16 q1, q2, q3
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vmult.i16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x09]
# CHECK-NOFP: vmult.i16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x09]
# CHECK: vmule.i16 q1, q2, q3 @ encoding: [0x14,0xef,0x56,0x29]
# CHECK-NOFP: vmule.i16 q1, q2, q3 @ encoding: [0x14,0xef,0x56,0x29]

vqrshl.u32 q0, r0
# CHECK: vqrshl.u32 q0, r0 @ encoding: [0x3b,0xfe,0xe0,0x1e]
# CHECK-NOFP: vqrshl.u32 q0, r0 @ encoding: [0x3b,0xfe,0xe0,0x1e]

vpste
vqrshlt.u16 q0, r0
vqrshle.s16 q0, q1, q2
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vqrshlt.u16 q0, r0 @ encoding: [0x37,0xfe,0xe0,0x1e]
# CHECK-NOFP: vqrshlt.u16 q0, r0 @ encoding: [0x37,0xfe,0xe0,0x1e]
# CHECK: vqrshle.s16 q0, q1, q2 @ encoding: [0x14,0xef,0x52,0x05]
# CHECK-NOFP: vqrshle.s16 q0, q1, q2 @ encoding: [0x14,0xef,0x52,0x05]

vpste
vrshlt.u16 q0, q1, q2
vrshle.s32 q0, r0
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vrshlt.u16 q0, q1, q2 @ encoding: [0x14,0xff,0x42,0x05]
# CHECK-NOFP: vrshlt.u16 q0, q1, q2 @ encoding: [0x14,0xff,0x42,0x05]
# CHECK: vrshle.s32 q0, r0 @ encoding: [0x3b,0xee,0x60,0x1e]
# CHECK-NOFP: vrshle.s32 q0, r0 @ encoding: [0x3b,0xee,0x60,0x1e]

vpste
vshlt.s8 q0, r0
vshle.u32 q0, r0
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vshlt.s8 q0, r0 @ encoding: [0x31,0xee,0x60,0x1e]
# CHECK-NOFP: vshlt.s8 q0, r0 @ encoding: [0x31,0xee,0x60,0x1e]
# CHECK: vshle.u32 q0, r0 @ encoding: [0x39,0xfe,0x60,0x1e]
# CHECK-NOFP: vshle.u32 q0, r0 @ encoding: [0x39,0xfe,0x60,0x1e]
