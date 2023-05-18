# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s

# CHECK: vshlc   q0, lr, #8  @ encoding: [0xa8,0xee,0xce,0x0f]
# CHECK-NOFP: vshlc   q0, lr, #8  @ encoding: [0xa8,0xee,0xce,0x0f]
vshlc   q0, lr, #8

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,32]
vshlc   q0, lr, #33

# CHECK: vmovlb.s8 q0, q6  @ encoding: [0xa8,0xee,0x4c,0x0f]
# CHECK-NOFP: vmovlb.s8 q0, q6  @ encoding: [0xa8,0xee,0x4c,0x0f]
vmovlb.s8 q0, q6

# CHECK: vmovlt.s8 q0, q4  @ encoding: [0xa8,0xee,0x48,0x1f]
# CHECK-NOFP: vmovlt.s8 q0, q4  @ encoding: [0xa8,0xee,0x48,0x1f]
vmovlt.s8 q0, q4

# CHECK: vpt.i8 eq, q0, q0
# CHECK-NOFP: vpt.i8 eq, q0, q0
# CHECK: vmovltt.s8 q0, q4  @ encoding: [0xa8,0xee,0x48,0x1f]
# CHECK-NOFP: vmovltt.s8 q0, q4  @ encoding: [0xa8,0xee,0x48,0x1f]
vpt.i8 eq, q0, q0
vmovltt.s8 q0, q4

# CHECK: vmovlb.u8 q0, q0  @ encoding: [0xa8,0xfe,0x40,0x0f]
# CHECK-NOFP: vmovlb.u8 q0, q0  @ encoding: [0xa8,0xfe,0x40,0x0f]
vmovlb.u8 q0, q0

# CHECK: vmovlt.u8 q0, q2  @ encoding: [0xa8,0xfe,0x44,0x1f]
# CHECK-NOFP: vmovlt.u8 q0, q2  @ encoding: [0xa8,0xfe,0x44,0x1f]
vmovlt.u8 q0, q2

# CHECK: vmovlb.u16 q1, q0  @ encoding: [0xb0,0xfe,0x40,0x2f]
# CHECK-NOFP: vmovlb.u16 q1, q0  @ encoding: [0xb0,0xfe,0x40,0x2f]
vmovlb.u16 q1, q0

# CHECK: vmovlt.u16 q0, q2  @ encoding: [0xb0,0xfe,0x44,0x1f]
# CHECK-NOFP: vmovlt.u16 q0, q2  @ encoding: [0xb0,0xfe,0x44,0x1f]
vmovlt.u16 q0, q2

# CHECK: vshllb.s8 q0, q2, #8  @ encoding: [0x31,0xee,0x05,0x0e]
# CHECK-NOFP: vshllb.s8 q0, q2, #8  @ encoding: [0x31,0xee,0x05,0x0e]
vshllb.s8 q0, q2, #8

# CHECK: vshllt.s8 q1, q5, #8  @ encoding: [0x31,0xee,0x0b,0x3e]
# CHECK-NOFP: vshllt.s8 q1, q5, #8  @ encoding: [0x31,0xee,0x0b,0x3e]
vshllt.s8 q1, q5, #8

# CHECK: vshllb.s8 q0, q0, #7  @ encoding: [0xaf,0xee,0x40,0x0f]
# CHECK-NOFP: vshllb.s8 q0, q0, #7  @ encoding: [0xaf,0xee,0x40,0x0f]
vshllb.s8 q0, q0, #7

# CHECK: vshllb.u8 q1, q1, #8  @ encoding: [0x31,0xfe,0x03,0x2e]
# CHECK-NOFP: vshllb.u8 q1, q1, #8  @ encoding: [0x31,0xfe,0x03,0x2e]
vshllb.u8 q1, q1, #8

# CHECK: vshllt.u8 q0, q0, #8  @ encoding: [0x31,0xfe,0x01,0x1e]
# CHECK-NOFP: vshllt.u8 q0, q0, #8  @ encoding: [0x31,0xfe,0x01,0x1e]
vshllt.u8 q0, q0, #8

# CHECK: vshllb.u8 q0, q0, #3  @ encoding: [0xab,0xfe,0x40,0x0f]
# CHECK-NOFP: vshllb.u8 q0, q0, #3  @ encoding: [0xab,0xfe,0x40,0x0f]
vshllb.u8 q0, q0, #3

# CHECK: vshllb.u16 q0, q5, #16  @ encoding: [0x35,0xfe,0x0b,0x0e]
# CHECK-NOFP: vshllb.u16 q0, q5, #16  @ encoding: [0x35,0xfe,0x0b,0x0e]
vshllb.u16 q0, q5, #16

# CHECK: vshllt.u16 q0, q3, #16  @ encoding: [0x35,0xfe,0x07,0x1e]
# CHECK-NOFP: vshllt.u16 q0, q3, #16  @ encoding: [0x35,0xfe,0x07,0x1e]
vshllt.u16 q0, q3, #16

# CHECK: vshllt.s16 q0, q0, #16  @ encoding: [0x35,0xee,0x01,0x1e]
# CHECK-NOFP: vshllt.s16 q0, q0, #16  @ encoding: [0x35,0xee,0x01,0x1e]
vshllt.s16 q0, q0, #16

# CHECK: vshllt.s16 q0, q0, #14  @ encoding: [0xbe,0xee,0x40,0x1f]
vshllt.s16 q0, q0, #14

# CHECK: vshllt.s16 q0, q0, #11  @ encoding: [0xbb,0xee,0x40,0x1f]
vshllt.s16 q0, q0, #11

# CHECK: vshllb.u16 q0, q2, #4  @ encoding: [0xb4,0xfe,0x44,0x0f]
# CHECK-NOFP: vshllb.u16 q0, q2, #4  @ encoding: [0xb4,0xfe,0x44,0x0f]
vshllb.u16 q0, q2, #4

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vshllb.s8 q0, q2, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vshllb.u8 q0, q2, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vshllb.u8 q0, q2, #0

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vshllb.s16 q0, q2, #17

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vshllb.u16 q0, q2, #17

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vshllb.u16 q0, q2, #0

# CHECK: vrshrnb.i16 q0, q3, #1  @ encoding: [0x8f,0xfe,0xc7,0x0f]
# CHECK-NOFP: vrshrnb.i16 q0, q3, #1  @ encoding: [0x8f,0xfe,0xc7,0x0f]
vrshrnb.i16 q0, q3, #1

# CHECK: vrshrnt.i16 q0, q2, #5  @ encoding: [0x8b,0xfe,0xc5,0x1f]
# CHECK-NOFP: vrshrnt.i16 q0, q2, #5  @ encoding: [0x8b,0xfe,0xc5,0x1f]
vrshrnt.i16 q0, q2, #5

# CHECK: vrshrnb.i32 q0, q4, #8  @ encoding: [0x98,0xfe,0xc9,0x0f]
# CHECK-NOFP: vrshrnb.i32 q0, q4, #8  @ encoding: [0x98,0xfe,0xc9,0x0f]
vrshrnb.i32 q0, q4, #8

# CHECK: vrshrnt.i32 q0, q2, #7  @ encoding: [0x99,0xfe,0xc5,0x1f]
# CHECK-NOFP: vrshrnt.i32 q0, q2, #7  @ encoding: [0x99,0xfe,0xc5,0x1f]
vrshrnt.i32 q0, q2, #7

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vrshrnb.i16 q0, q3, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vrshrnb.i32 q0, q3, #17

# CHECK: vshrnb.i16 q1, q2, #1  @ encoding: [0x8f,0xee,0xc5,0x2f]
# CHECK-NOFP: vshrnb.i16 q1, q2, #1  @ encoding: [0x8f,0xee,0xc5,0x2f]
vshrnb.i16 q1, q2, #1

# CHECK: vshrnt.i16 q0, q1, #1  @ encoding: [0x8f,0xee,0xc3,0x1f]
# CHECK-NOFP: vshrnt.i16 q0, q1, #1  @ encoding: [0x8f,0xee,0xc3,0x1f]
vshrnt.i16 q0, q1, #1

# CHECK: vshrnb.i32 q0, q0, #12  @ encoding: [0x94,0xee,0xc1,0x0f]
# CHECK-NOFP: vshrnb.i32 q0, q0, #12  @ encoding: [0x94,0xee,0xc1,0x0f]
vshrnb.i32 q0, q0, #12

# CHECK: vshrnt.i32 q0, q2, #4  @ encoding: [0x9c,0xee,0xc5,0x1f]
# CHECK-NOFP: vshrnt.i32 q0, q2, #4  @ encoding: [0x9c,0xee,0xc5,0x1f]
vshrnt.i32 q0, q2, #4

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vshrnb.i16 q1, q2, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vshrnb.i32 q1, q2, #17

# CHECK: vqrshrunb.s16 q0, q2, #8  @ encoding: [0x88,0xfe,0xc4,0x0f]
# CHECK-NOFP: vqrshrunb.s16 q0, q2, #8  @ encoding: [0x88,0xfe,0xc4,0x0f]
vqrshrunb.s16 q0, q2, #8

# CHECK: vqrshrunt.s16 q0, q0, #6  @ encoding: [0x8a,0xfe,0xc0,0x1f]
# CHECK-NOFP: vqrshrunt.s16 q0, q0, #6  @ encoding: [0x8a,0xfe,0xc0,0x1f]
vqrshrunt.s16 q0, q0, #6

# CHECK: vqrshrunt.s32 q0, q1, #8  @ encoding: [0x98,0xfe,0xc2,0x1f]
# CHECK-NOFP: vqrshrunt.s32 q0, q1, #8  @ encoding: [0x98,0xfe,0xc2,0x1f]
vqrshrunt.s32 q0, q1, #8

# CHECK: vqrshrunb.s32 q0, q7, #13  @ encoding: [0x93,0xfe,0xce,0x0f]
# CHECK-NOFP: vqrshrunb.s32 q0, q7, #13  @ encoding: [0x93,0xfe,0xce,0x0f]
vqrshrunb.s32 q0, q7, #13

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vqrshrunb.s16 q0, q2, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vqrshrunb.s32 q0, q2, #17

# CHECK: vqshrunb.s16 q0, q7, #5  @ encoding: [0x8b,0xee,0xce,0x0f]
# CHECK-NOFP: vqshrunb.s16 q0, q7, #5  @ encoding: [0x8b,0xee,0xce,0x0f]
vqshrunb.s16 q0, q7, #5

# CHECK: vqshrunt.s16 q0, q1, #7  @ encoding: [0x89,0xee,0xc2,0x1f]
# CHECK-NOFP: vqshrunt.s16 q0, q1, #7  @ encoding: [0x89,0xee,0xc2,0x1f]
vqshrunt.s16 q0, q1, #7

# CHECK: vqshrunb.s32 q0, q6, #4  @ encoding: [0x9c,0xee,0xcc,0x0f]
# CHECK-NOFP: vqshrunb.s32 q0, q6, #4  @ encoding: [0x9c,0xee,0xcc,0x0f]
vqshrunb.s32 q0, q6, #4

# CHECK: vqshrunt.s32 q0, q2, #10  @ encoding: [0x96,0xee,0xc4,0x1f]
# CHECK-NOFP: vqshrunt.s32 q0, q2, #10  @ encoding: [0x96,0xee,0xc4,0x1f]
vqshrunt.s32 q0, q2, #10

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vqshrunt.s16 q0, q1, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vqshrunb.s32 q0, q6, #17

# CHECK: vqrshrnb.s16 q0, q7, #8  @ encoding: [0x88,0xee,0x4f,0x0f]
# CHECK-NOFP: vqrshrnb.s16 q0, q7, #8  @ encoding: [0x88,0xee,0x4f,0x0f]
vqrshrnb.s16 q0, q7, #8

# CHECK: vqrshrnt.u16 q1, q3, #4  @ encoding: [0x8c,0xfe,0x47,0x3f]
# CHECK-NOFP: vqrshrnt.u16 q1, q3, #4  @ encoding: [0x8c,0xfe,0x47,0x3f]
vqrshrnt.u16 q1, q3, #4

# CHECK: vqrshrnb.u32 q0, q1, #7  @ encoding: [0x99,0xfe,0x43,0x0f]
# CHECK-NOFP: vqrshrnb.u32 q0, q1, #7  @ encoding: [0x99,0xfe,0x43,0x0f]
vqrshrnb.u32 q0, q1, #7

# CHECK: vqrshrnt.s32 q0, q1, #11  @ encoding: [0x95,0xee,0x43,0x1f]
# CHECK-NOFP: vqrshrnt.s32 q0, q1, #11  @ encoding: [0x95,0xee,0x43,0x1f]
vqrshrnt.s32 q0, q1, #11

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vqrshrnb.s16 q0, q7, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vqrshrnb.s32 q0, q7, #17

# CHECK: vqshrnb.s16 q0, q6, #5  @ encoding: [0x8b,0xee,0x4c,0x0f]
# CHECK-NOFP: vqshrnb.s16 q0, q6, #5  @ encoding: [0x8b,0xee,0x4c,0x0f]
vqshrnb.s16 q0, q6, #5

# CHECK: vqshrnt.s16 q0, q1, #4  @ encoding: [0x8c,0xee,0x42,0x1f]
# CHECK-NOFP: vqshrnt.s16 q0, q1, #4  @ encoding: [0x8c,0xee,0x42,0x1f]
vqshrnt.s16 q0, q1, #4

# CHECK: vqshrnb.u16 q0, q3, #7  @ encoding: [0x89,0xfe,0x46,0x0f]
# CHECK-NOFP: vqshrnb.u16 q0, q3, #7  @ encoding: [0x89,0xfe,0x46,0x0f]
vqshrnb.u16 q0, q3, #7

# CHECK: vqshrnt.u16 q0, q2, #8  @ encoding: [0x88,0xfe,0x44,0x1f]
# CHECK-NOFP: vqshrnt.u16 q0, q2, #8  @ encoding: [0x88,0xfe,0x44,0x1f]
vqshrnt.u16 q0, q2, #8

# CHECK: vqshrnt.s32 q1, q4, #3  @ encoding: [0x9d,0xee,0x48,0x3f]
# CHECK-NOFP: vqshrnt.s32 q1, q4, #3  @ encoding: [0x9d,0xee,0x48,0x3f]
vqshrnt.s32 q1, q4, #3

# CHECK: vqshrnb.u32 q0, q2, #14  @ encoding: [0x92,0xfe,0x44,0x0f]
# CHECK-NOFP: vqshrnb.u32 q0, q2, #14  @ encoding: [0x92,0xfe,0x44,0x0f]
vqshrnb.u32 q0, q2, #14

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vqshrnb.s16 q0, q6, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vqshrnb.u32 q0, q6, #17

# CHECK: vshl.s8 q6, q6, q6  @ encoding: [0x0c,0xef,0x4c,0xc4]
# CHECK-NOFP: vshl.s8 q6, q6, q6  @ encoding: [0x0c,0xef,0x4c,0xc4]
vshl.s8 q6, q6, q6

# CHECK: vshl.s16 q0, q4, q2  @ encoding: [0x14,0xef,0x48,0x04]
# CHECK-NOFP: vshl.s16 q0, q4, q2  @ encoding: [0x14,0xef,0x48,0x04]
vshl.s16 q0, q4, q2

# CHECK: vshl.s32 q1, q1, q5  @ encoding: [0x2a,0xef,0x42,0x24]
# CHECK-NOFP: vshl.s32 q1, q1, q5  @ encoding: [0x2a,0xef,0x42,0x24]
vshl.s32 q1, q1, q5

# CHECK: vshl.u8 q1, q7, q2  @ encoding: [0x04,0xff,0x4e,0x24]
# CHECK-NOFP: vshl.u8 q1, q7, q2  @ encoding: [0x04,0xff,0x4e,0x24]
vshl.u8 q1, q7, q2

# CHECK: vshl.u16 q0, q4, q0  @ encoding: [0x10,0xff,0x48,0x04]
# CHECK-NOFP: vshl.u16 q0, q4, q0  @ encoding: [0x10,0xff,0x48,0x04]
vshl.u16 q0, q4, q0

# CHECK: vshl.u32 q2, q2, q4  @ encoding: [0x28,0xff,0x44,0x44]
# CHECK-NOFP: vshl.u32 q2, q2, q4  @ encoding: [0x28,0xff,0x44,0x44]
vshl.u32 q2, q2, q4

# CHECK: vqshl.s8 q0, q1, q6  @ encoding: [0x0c,0xef,0x52,0x04]
# CHECK-NOFP: vqshl.s8 q0, q1, q6  @ encoding: [0x0c,0xef,0x52,0x04]
vqshl.s8 q0, q1, q6

# CHECK: vqshl.s16 q4, q3, q7  @ encoding: [0x1e,0xef,0x56,0x84]
# CHECK-NOFP: vqshl.s16 q4, q3, q7  @ encoding: [0x1e,0xef,0x56,0x84]
vqshl.s16 q4, q3, q7

# CHECK: vqshl.s32 q0, q5, q5  @ encoding: [0x2a,0xef,0x5a,0x04]
# CHECK-NOFP: vqshl.s32 q0, q5, q5  @ encoding: [0x2a,0xef,0x5a,0x04]
vqshl.s32 q0, q5, q5

# CHECK: vqshl.u8 q0, q0, q6  @ encoding: [0x0c,0xff,0x50,0x04]
# CHECK-NOFP: vqshl.u8 q0, q0, q6  @ encoding: [0x0c,0xff,0x50,0x04]
vqshl.u8 q0, q0, q6

# CHECK: vqshl.u16 q0, q5, q4  @ encoding: [0x18,0xff,0x5a,0x04]
# CHECK-NOFP: vqshl.u16 q0, q5, q4  @ encoding: [0x18,0xff,0x5a,0x04]
vqshl.u16 q0, q5, q4

# CHECK: vqshl.u32 q1, q0, q4  @ encoding: [0x28,0xff,0x50,0x24]
# CHECK-NOFP: vqshl.u32 q1, q0, q4  @ encoding: [0x28,0xff,0x50,0x24]
vqshl.u32 q1, q0, q4

# CHECK: vqrshl.s8 q1, q6, q1  @ encoding: [0x02,0xef,0x5c,0x25]
# CHECK-NOFP: vqrshl.s8 q1, q6, q1  @ encoding: [0x02,0xef,0x5c,0x25]
vqrshl.s8 q1, q6, q1

# CHECK: vqrshl.s16 q2, q4, q6  @ encoding: [0x1c,0xef,0x58,0x45]
# CHECK-NOFP: vqrshl.s16 q2, q4, q6  @ encoding: [0x1c,0xef,0x58,0x45]
vqrshl.s16 q2, q4, q6

# CHECK: vqrshl.s32 q0, q0, q5  @ encoding: [0x2a,0xef,0x50,0x05]
# CHECK-NOFP: vqrshl.s32 q0, q0, q5  @ encoding: [0x2a,0xef,0x50,0x05]
vqrshl.s32 q0, q0, q5

# CHECK: vqrshl.u8 q0, q2, q1  @ encoding: [0x02,0xff,0x54,0x05]
# CHECK-NOFP: vqrshl.u8 q0, q2, q1  @ encoding: [0x02,0xff,0x54,0x05]
vqrshl.u8 q0, q2, q1

# CHECK: vqrshl.u16 q1, q6, q0  @ encoding: [0x10,0xff,0x5c,0x25]
# CHECK-NOFP: vqrshl.u16 q1, q6, q0  @ encoding: [0x10,0xff,0x5c,0x25]
vqrshl.u16 q1, q6, q0

# CHECK: vqrshl.u32 q0, q0, q0  @ encoding: [0x20,0xff,0x50,0x05]
# CHECK-NOFP: vqrshl.u32 q0, q0, q0  @ encoding: [0x20,0xff,0x50,0x05]
vqrshl.u32 q0, q0, q0

# CHECK: vrshl.s8 q0, q6, q4  @ encoding: [0x08,0xef,0x4c,0x05]
# CHECK-NOFP: vrshl.s8 q0, q6, q4  @ encoding: [0x08,0xef,0x4c,0x05]
vrshl.s8 q0, q6, q4

# CHECK: vrshl.s16 q1, q4, q7  @ encoding: [0x1e,0xef,0x48,0x25]
# CHECK-NOFP: vrshl.s16 q1, q4, q7  @ encoding: [0x1e,0xef,0x48,0x25]
vrshl.s16 q1, q4, q7

# CHECK: vrshl.s32 q1, q4, q4  @ encoding: [0x28,0xef,0x48,0x25]
# CHECK-NOFP: vrshl.s32 q1, q4, q4  @ encoding: [0x28,0xef,0x48,0x25]
vrshl.s32 q1, q4, q4

# CHECK: vrshl.u8 q0, q3, q5  @ encoding: [0x0a,0xff,0x46,0x05]
# CHECK-NOFP: vrshl.u8 q0, q3, q5  @ encoding: [0x0a,0xff,0x46,0x05]
vrshl.u8 q0, q3, q5

# CHECK: vrshl.u16 q5, q6, q5  @ encoding: [0x1a,0xff,0x4c,0xa5]
# CHECK-NOFP: vrshl.u16 q5, q6, q5  @ encoding: [0x1a,0xff,0x4c,0xa5]
vrshl.u16 q5, q6, q5

# CHECK: vrshl.u32 q1, q7, q3  @ encoding: [0x26,0xff,0x4e,0x25]
# CHECK-NOFP: vrshl.u32 q1, q7, q3  @ encoding: [0x26,0xff,0x4e,0x25]
vrshl.u32 q1, q7, q3

# CHECK: vsri.8 q0, q2, #3  @ encoding: [0x8d,0xff,0x54,0x04]
# CHECK-NOFP: vsri.8 q0, q2, #3  @ encoding: [0x8d,0xff,0x54,0x04]
vsri.8 q0, q2, #3

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vsri.8 q0, q2, #9

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vsri.8 q0, q2, #0

# CHECK: vsri.16 q0, q2, #5  @ encoding: [0x9b,0xff,0x54,0x04]
# CHECK-NOFP: vsri.16 q0, q2, #5  @ encoding: [0x9b,0xff,0x54,0x04]
vsri.16 q0, q2, #5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vsri.16 q0, q2, #17

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vsri.16 q0, q2, #0

# CHECK: vsri.32 q0, q1, #15  @ encoding: [0xb1,0xff,0x52,0x04]
# CHECK-NOFP: vsri.32 q0, q1, #15  @ encoding: [0xb1,0xff,0x52,0x04]
vsri.32 q0, q1, #15

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,32]
vsri.32 q0, q2, #33

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,32]
vsri.32 q0, q2, #0

# CHECK: vsli.8 q0, q3, #3  @ encoding: [0x8b,0xff,0x56,0x05]
# CHECK-NOFP: vsli.8 q0, q3, #3  @ encoding: [0x8b,0xff,0x56,0x05]
vsli.8 q0, q3, #3

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,7]
vsli.8 q0, q3, #8

# CHECK: vsli.16 q0, q1, #12  @ encoding: [0x9c,0xff,0x52,0x05]
# CHECK-NOFP: vsli.16 q0, q1, #12  @ encoding: [0x9c,0xff,0x52,0x05]
vsli.16 q0, q1, #12

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,15]
vsli.16 q0, q3, #16

# CHECK: vsli.32 q0, q1, #8  @ encoding: [0xa8,0xff,0x52,0x05]
# CHECK-NOFP: vsli.32 q0, q1, #8  @ encoding: [0xa8,0xff,0x52,0x05]
vsli.32 q0, q1, #8

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,31]
vsli.32 q0, q1, #32

# CHECK: vqshl.s8 q0, q4, #6  @ encoding: [0x8e,0xef,0x58,0x07]
# CHECK-NOFP: vqshl.s8 q0, q4, #6  @ encoding: [0x8e,0xef,0x58,0x07]
vqshl.s8 q0, q4, #6

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,7]
vqshl.s8 q0, q4, #8

# CHECK: vqshl.u8 q0, q6, #6  @ encoding: [0x8e,0xff,0x5c,0x07]
# CHECK-NOFP: vqshl.u8 q0, q6, #6  @ encoding: [0x8e,0xff,0x5c,0x07]
vqshl.u8 q0, q6, #6

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,7]
vqshl.u8 q0, q4, #8

# CHECK: vqshl.s16 q1, q2, #5  @ encoding: [0x95,0xef,0x54,0x27]
# CHECK-NOFP: vqshl.s16 q1, q2, #5  @ encoding: [0x95,0xef,0x54,0x27]
vqshl.s16 q1, q2, #5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,15]
vqshl.s16 q1, q2, #16

# CHECK: vqshl.u16 q0, q5, #3  @ encoding: [0x93,0xff,0x5a,0x07]
# CHECK-NOFP: vqshl.u16 q0, q5, #3  @ encoding: [0x93,0xff,0x5a,0x07]
vqshl.u16 q0, q5, #3

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,15]
vqshl.u16 q1, q2, #16

# CHECK: vqshl.s32 q1, q3, #29  @ encoding: [0xbd,0xef,0x56,0x27]
# CHECK-NOFP: vqshl.s32 q1, q3, #29  @ encoding: [0xbd,0xef,0x56,0x27]
vqshl.s32 q1, q3, #29

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,31]
vqshl.s32 q1, q3, #32

# CHECK: vqshl.u32 q0, q2, #19  @ encoding: [0xb3,0xff,0x54,0x07]
# CHECK-NOFP: vqshl.u32 q0, q2, #19  @ encoding: [0xb3,0xff,0x54,0x07]
vqshl.u32 q0, q2, #19

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,31]
vqshl.u32 q0, q2, #32

# CHECK: vqshlu.s8 q0, q1, #0  @ encoding: [0x88,0xff,0x52,0x06]
# CHECK-NOFP: vqshlu.s8 q0, q1, #0  @ encoding: [0x88,0xff,0x52,0x06]
vqshlu.s8 q0, q1, #0

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,7]
vqshlu.s8 q0, q1, #8

# CHECK: vqshlu.s16 q2, q1, #12  @ encoding: [0x9c,0xff,0x52,0x46]
# CHECK-NOFP: vqshlu.s16 q2, q1, #12  @ encoding: [0x9c,0xff,0x52,0x46]
vqshlu.s16 q2, q1, #12

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,15]
vqshlu.s16 q0, q1, #16

# CHECK: vqshlu.s32 q0, q4, #26  @ encoding: [0xba,0xff,0x58,0x06]
# CHECK-NOFP: vqshlu.s32 q0, q4, #26  @ encoding: [0xba,0xff,0x58,0x06]
vqshlu.s32 q0, q4, #26

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,31]
vqshlu.s32 q0, q1, #32

# CHECK: vrshr.s8 q1, q3, #7  @ encoding: [0x89,0xef,0x56,0x22]
# CHECK-NOFP: vrshr.s8 q1, q3, #7  @ encoding: [0x89,0xef,0x56,0x22]
vrshr.s8 q1, q3, #7

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vrshr.s8 q1, q3, #9

# CHECK: vrshr.u8 q1, q3, #2  @ encoding: [0x8e,0xff,0x56,0x22]
# CHECK-NOFP: vrshr.u8 q1, q3, #2  @ encoding: [0x8e,0xff,0x56,0x22]
vrshr.u8 q1, q3, #2

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vrshr.u8 q1, q3, #9

# CHECK: vrshr.s16 q0, q1, #10  @ encoding: [0x96,0xef,0x52,0x02]
# CHECK-NOFP: vrshr.s16 q0, q1, #10  @ encoding: [0x96,0xef,0x52,0x02]
vrshr.s16 q0, q1, #10

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vrshr.s16 q0, q1, #17

# CHECK: vrshr.u16 q0, q5, #12  @ encoding: [0x94,0xff,0x5a,0x02]
# CHECK-NOFP: vrshr.u16 q0, q5, #12  @ encoding: [0x94,0xff,0x5a,0x02]
vrshr.u16 q0, q5, #12

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vrshr.u16 q0, q5, #20

# CHECK: vrshr.s32 q0, q5, #23  @ encoding: [0xa9,0xef,0x5a,0x02]
# CHECK-NOFP: vrshr.s32 q0, q5, #23  @ encoding: [0xa9,0xef,0x5a,0x02]
vrshr.s32 q0, q5, #23

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,32]
vrshr.s32 q0, q5, #33

# CHECK: vrshr.u32 q0, q1, #30  @ encoding: [0xa2,0xff,0x52,0x02]
# CHECK-NOFP: vrshr.u32 q0, q1, #30  @ encoding: [0xa2,0xff,0x52,0x02]
vrshr.u32 q0, q1, #30

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,32]
vrshr.u32 q0, q1, #55

# CHECK: vshr.s8 q0, q7, #4  @ encoding: [0x8c,0xef,0x5e,0x00]
# CHECK-NOFP: vshr.s8 q0, q7, #4  @ encoding: [0x8c,0xef,0x5e,0x00]
vshr.s8 q0, q7, #4

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vshr.s8 q0, q7, #9

# CHECK: vshr.u8 q0, q2, #5  @ encoding: [0x8b,0xff,0x54,0x00]
# CHECK-NOFP: vshr.u8 q0, q2, #5  @ encoding: [0x8b,0xff,0x54,0x00]
vshr.u8 q0, q2, #5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,8]
vshr.u8 q0, q2, #9

# CHECK: vshr.s16 q0, q3, #16  @ encoding: [0x90,0xef,0x56,0x00]
# CHECK-NOFP: vshr.s16 q0, q3, #16  @ encoding: [0x90,0xef,0x56,0x00]
vshr.s16 q0, q3, #16

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vshr.s16 q0, q2, #17

# CHECK: vshr.u16 q7, q6, #8  @ encoding: [0x98,0xff,0x5c,0xe0]
# CHECK-NOFP: vshr.u16 q7, q6, #8  @ encoding: [0x98,0xff,0x5c,0xe0]
vshr.u16 q7, q6, #8

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,16]
vshr.u16 q7, q6, #20

# CHECK: vshr.s32 q0, q6, #24  @ encoding: [0xa8,0xef,0x5c,0x00]
# CHECK-NOFP: vshr.s32 q0, q6, #24  @ encoding: [0xa8,0xef,0x5c,0x00]
vshr.s32 q0, q6, #24

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,32]
vshr.s32 q0, q6, #33

# CHECK: vshr.u32 q2, q5, #30  @ encoding: [0xa2,0xff,0x5a,0x40]
# CHECK-NOFP: vshr.u32 q2, q5, #30  @ encoding: [0xa2,0xff,0x5a,0x40]
vshr.u32 q2, q5, #30

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,32]
vshr.u32 q2, q5, #33

# CHECK: vshl.i8 q0, q6, #6  @ encoding: [0x8e,0xef,0x5c,0x05]
# CHECK-NOFP: vshl.i8 q0, q6, #6  @ encoding: [0x8e,0xef,0x5c,0x05]
vshl.i8 q0, q6, #6

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,7]
vshl.i8 q0, q6, #8

# CHECK: vshl.i16 q1, q0, #12  @ encoding: [0x9c,0xef,0x50,0x25]
# CHECK-NOFP: vshl.i16 q1, q0, #12  @ encoding: [0x9c,0xef,0x50,0x25]
vshl.i16 q1, q0, #12

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,15]
vshl.i16 q1, q0, #16

# CHECK: vshl.i32 q2, q2, #26  @ encoding: [0xba,0xef,0x54,0x45]
# CHECK-NOFP: vshl.i32 q2, q2, #26  @ encoding: [0xba,0xef,0x54,0x45]
vshl.i32 q2, q2, #26

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [0,31]
vshl.i32 q2, q2, #33

vshllt.s8 q0, q1, #1
# CHECK: vshllt.s8 q0, q1, #1 @ encoding: [0xa9,0xee,0x42,0x1f]
# CHECK-NOFP: vshllt.s8 q0, q1, #1 @ encoding: [0xa9,0xee,0x42,0x1f]

vpste
vshlltt.s16 q0, q1, #4
vshllbe.u16 q0, q1, #8
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vshlltt.s16 q0, q1, #4 @ encoding: [0xb4,0xee,0x42,0x1f]
# CHECK-NOFP: vshlltt.s16 q0, q1, #4 @ encoding: [0xb4,0xee,0x42,0x1f]
# CHECK: vshllbe.u16 q0, q1, #8 @ encoding: [0xb8,0xfe,0x42,0x0f]
# CHECK-NOFP: vshllbe.u16 q0, q1, #8 @ encoding: [0xb8,0xfe,0x42,0x0f]
