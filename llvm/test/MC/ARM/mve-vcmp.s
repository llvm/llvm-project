# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: FileCheck --check-prefix=ERROR-NOFP < %t %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN: FileCheck --check-prefix=ERROR < %t %s

# CHECK: vcmp.f16 eq, q0, q4  @ encoding: [0x31,0xfe,0x08,0x0f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f16 eq, q0, q4  @ encoding: [0x31,0xfe,0x08,0x0f]
vcmp.f16 eq, q0, q4

# CHECK: vcmp.f16 ne, q2, q7  @ encoding: [0x35,0xfe,0x8e,0x0f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f16 ne, q2, q7  @ encoding: [0x35,0xfe,0x8e,0x0f]
vcmp.f16 ne, q2, q7

# CHECK: vcmp.f16 ge, q0, q0  @ encoding: [0x31,0xfe,0x00,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f16 ge, q0, q0  @ encoding: [0x31,0xfe,0x00,0x1f]
vcmp.f16 ge, q0, q0

# CHECK: vcmp.f16 lt, q0, q1  @ encoding: [0x31,0xfe,0x82,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f16 lt, q0, q1  @ encoding: [0x31,0xfe,0x82,0x1f]
vcmp.f16 lt, q0, q1

# CHECK: vcmp.f16 gt, q1, q4  @ encoding: [0x33,0xfe,0x09,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f16 gt, q1, q4  @ encoding: [0x33,0xfe,0x09,0x1f]
vcmp.f16 gt, q1, q4

# CHECK: vcmp.f16 le, q2, q6  @ encoding: [0x35,0xfe,0x8d,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f16 le, q2, q6  @ encoding: [0x35,0xfe,0x8d,0x1f]
vcmp.f16 le, q2, q6

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for floating-point comparison must be EQ, NE, LT, GT, LE or GE
vcmp.f16 hi, q2, q6

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for floating-point comparison must be EQ, NE, LT, GT, LE or GE
vcmp.f16 hs, q2, q6

# CHECK: vcmp.f32 eq, q2, q5  @ encoding: [0x35,0xee,0x0a,0x0f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f32 eq, q2, q5  @ encoding: [0x35,0xee,0x0a,0x0f]
vcmp.f32 eq, q2, q5

# CHECK: vcmp.f32 ne, q3, q4  @ encoding: [0x37,0xee,0x88,0x0f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f32 ne, q3, q4  @ encoding: [0x37,0xee,0x88,0x0f]
vcmp.f32 ne, q3, q4

# CHECK: vcmp.f32 ge, q0, q7  @ encoding: [0x31,0xee,0x0e,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f32 ge, q0, q7  @ encoding: [0x31,0xee,0x0e,0x1f]
vcmp.f32 ge, q0, q7

# CHECK: vcmp.f32 lt, q5, q2  @ encoding: [0x3b,0xee,0x84,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f32 lt, q5, q2  @ encoding: [0x3b,0xee,0x84,0x1f]
vcmp.f32 lt, q5, q2

# CHECK: vcmp.f32 gt, q2, q7  @ encoding: [0x35,0xee,0x0f,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f32 gt, q2, q7  @ encoding: [0x35,0xee,0x0f,0x1f]
vcmp.f32 gt, q2, q7

# CHECK: vcmp.f32 le, q2, q4  @ encoding: [0x35,0xee,0x89,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f32 le, q2, q4  @ encoding: [0x35,0xee,0x89,0x1f]
vcmp.f32 le, q2, q4

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for floating-point comparison must be EQ, NE, LT, GT, LE or GE
vcmp.f32 hi, q2, q6

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for floating-point comparison must be EQ, NE, LT, GT, LE or GE
vcmp.f32 hs, q2, q6

# CHECK: vcmp.i8 eq, q4, q6  @ encoding: [0x09,0xfe,0x0c,0x0f]
# CHECK-NOFP: vcmp.i8 eq, q4, q6  @ encoding: [0x09,0xfe,0x0c,0x0f]
vcmp.i8 eq, q4, q6

# CHECK: vcmp.i8 ne, q2, q2  @ encoding: [0x05,0xfe,0x84,0x0f]
# CHECK-NOFP: vcmp.i8 ne, q2, q2  @ encoding: [0x05,0xfe,0x84,0x0f]
vcmp.i8 ne, q2, q2

# ERROR: [[@LINE+1]]:9: {{error|note}}: condition code for sign-independent integer comparison must be EQ or NE
vcmp.i8 hs, q2, q6

# ERROR: [[@LINE+1]]:9: {{error|note}}: condition code for sign-independent integer comparison must be EQ or NE
vcmp.i8 le, q2, q6

# CHECK: vcmp.i8 eq, q4, q6  @ encoding: [0x09,0xfe,0x0c,0x0f]
# CHECK-NOFP: vcmp.i8 eq, q4, q6  @ encoding: [0x09,0xfe,0x0c,0x0f]
vcmp.s8 eq, q4, q6

# CHECK: vcmp.i8 ne, q2, q2  @ encoding: [0x05,0xfe,0x84,0x0f]
# CHECK-NOFP: vcmp.i8 ne, q2, q2  @ encoding: [0x05,0xfe,0x84,0x0f]
vcmp.s8 ne, q2, q2

# CHECK: vcmp.i8 eq, q4, q6  @ encoding: [0x09,0xfe,0x0c,0x0f]
# CHECK-NOFP: vcmp.i8 eq, q4, q6  @ encoding: [0x09,0xfe,0x0c,0x0f]
vcmp.u8 eq, q4, q6

# CHECK: vcmp.i8 ne, q2, q2  @ encoding: [0x05,0xfe,0x84,0x0f]
# CHECK-NOFP: vcmp.i8 ne, q2, q2  @ encoding: [0x05,0xfe,0x84,0x0f]
vcmp.u8 ne, q2, q2

# CHECK: vcmp.s8 ge, q0, q0  @ encoding: [0x01,0xfe,0x00,0x1f]
# CHECK-NOFP: vcmp.s8 ge, q0, q0  @ encoding: [0x01,0xfe,0x00,0x1f]
vcmp.s8 ge, q0, q0

# CHECK: vcmp.s8 lt, q2, q7  @ encoding: [0x05,0xfe,0x8e,0x1f]
# CHECK-NOFP: vcmp.s8 lt, q2, q7  @ encoding: [0x05,0xfe,0x8e,0x1f]
vcmp.s8 lt, q2, q7

# CHECK: vcmp.s8 gt, q4, q3  @ encoding: [0x09,0xfe,0x07,0x1f]
# CHECK-NOFP: vcmp.s8 gt, q4, q3  @ encoding: [0x09,0xfe,0x07,0x1f]
vcmp.s8 gt, q4, q3

# CHECK: vcmp.s8 le, q7, q3  @ encoding: [0x0f,0xfe,0x87,0x1f]
# CHECK-NOFP: vcmp.s8 le, q7, q3  @ encoding: [0x0f,0xfe,0x87,0x1f]
vcmp.s8 le, q7, q3

# ERROR: [[@LINE+1]]:9: {{error|note}}: condition code for signed integer comparison must be EQ, NE, LT, GT, LE or GE
vcmp.s8 hs, q2, q6

# CHECK: vcmp.u8 hi, q1, q4  @ encoding: [0x03,0xfe,0x89,0x0f]
# CHECK-NOFP: vcmp.u8 hi, q1, q4  @ encoding: [0x03,0xfe,0x89,0x0f]
vcmp.u8 hi, q1, q4

# CHECK: vcmp.u8 cs, q1, q4  @ encoding: [0x03,0xfe,0x09,0x0f]
# CHECK-NOFP: vcmp.u8 cs, q1, q4  @ encoding: [0x03,0xfe,0x09,0x0f]
vcmp.u8 cs, q1, q4

# ERROR: [[@LINE+1]]:9: {{error|note}}: condition code for unsigned integer comparison must be EQ, NE, HS or HI
vcmp.u8 gt, q2, q6

# ERROR: [[@LINE+1]]:9: {{error|note}}: condition code for unsigned integer comparison must be EQ, NE, HS or HI
vcmp.u8 lo, q2, q6

# CHECK: vcmp.i16 eq, q4, q7  @ encoding: [0x19,0xfe,0x0e,0x0f]
# CHECK-NOFP: vcmp.i16 eq, q4, q7  @ encoding: [0x19,0xfe,0x0e,0x0f]
vcmp.i16 eq, q4, q7

# CHECK: vcmp.i16 ne, q2, q1  @ encoding: [0x15,0xfe,0x82,0x0f]
# CHECK-NOFP: vcmp.i16 ne, q2, q1  @ encoding: [0x15,0xfe,0x82,0x0f]
vcmp.i16 ne, q2, q1

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for sign-independent integer comparison must be EQ or NE
vcmp.i16 hi, q2, q6

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for sign-independent integer comparison must be EQ or NE
vcmp.i16 lt, q2, q6

# CHECK: vcmp.s16 ge, q1, q7  @ encoding: [0x13,0xfe,0x0e,0x1f]
# CHECK-NOFP: vcmp.s16 ge, q1, q7  @ encoding: [0x13,0xfe,0x0e,0x1f]
vcmp.s16 ge, q1, q7

# CHECK: vcmp.s16 lt, q0, q1  @ encoding: [0x11,0xfe,0x82,0x1f]
# CHECK-NOFP: vcmp.s16 lt, q0, q1  @ encoding: [0x11,0xfe,0x82,0x1f]
vcmp.s16 lt, q0, q1

# CHECK: vcmp.s16 gt, q1, q7  @ encoding: [0x13,0xfe,0x0f,0x1f]
# CHECK-NOFP: vcmp.s16 gt, q1, q7  @ encoding: [0x13,0xfe,0x0f,0x1f]
vcmp.s16 gt, q1, q7

# CHECK: vcmp.s16 le, q2, q1  @ encoding: [0x15,0xfe,0x83,0x1f]
# CHECK-NOFP: vcmp.s16 le, q2, q1  @ encoding: [0x15,0xfe,0x83,0x1f]
vcmp.s16 le, q2, q1

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for signed integer comparison must be EQ, NE, LT, GT, LE or GE
vcmp.s16 hi, q2, q6

# CHECK: vcmp.u16 hi, q1, q4  @ encoding: [0x13,0xfe,0x89,0x0f]
# CHECK-NOFP: vcmp.u16 hi, q1, q4  @ encoding: [0x13,0xfe,0x89,0x0f]
vcmp.u16 hi, q1, q4

# CHECK: vcmp.u16 cs, q1, q4  @ encoding: [0x13,0xfe,0x09,0x0f]
# CHECK-NOFP: vcmp.u16 cs, q1, q4  @ encoding: [0x13,0xfe,0x09,0x0f]
vcmp.u16 cs, q1, q4

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for unsigned integer comparison must be EQ, NE, HS or HI
vcmp.u16 ge, q2, q6

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for unsigned integer comparison must be EQ, NE, HS or HI
vcmp.u16 ls, q2, q6

# CHECK: vcmp.i32 eq, q2, q7  @ encoding: [0x25,0xfe,0x0e,0x0f]
# CHECK-NOFP: vcmp.i32 eq, q2, q7  @ encoding: [0x25,0xfe,0x0e,0x0f]
vcmp.i32 eq, q2, q7

# CHECK: vcmp.i32 ne, q2, q4  @ encoding: [0x25,0xfe,0x88,0x0f]
# CHECK-NOFP: vcmp.i32 ne, q2, q4  @ encoding: [0x25,0xfe,0x88,0x0f]
vcmp.i32 ne, q2, q4

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for sign-independent integer comparison must be EQ or NE
vcmp.i32 lo, q2, q6

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for sign-independent integer comparison must be EQ or NE
vcmp.i32 ge, q2, q6

# CHECK: vcmp.s32 ge, q5, q5  @ encoding: [0x2b,0xfe,0x0a,0x1f]
# CHECK-NOFP: vcmp.s32 ge, q5, q5  @ encoding: [0x2b,0xfe,0x0a,0x1f]
vcmp.s32 ge, q5, q5

# CHECK: vcmp.s32 lt, q2, q2  @ encoding: [0x25,0xfe,0x84,0x1f]
# CHECK-NOFP: vcmp.s32 lt, q2, q2  @ encoding: [0x25,0xfe,0x84,0x1f]
vcmp.s32 lt, q2, q2

# CHECK: vcmp.s32 gt, q0, q1  @ encoding: [0x21,0xfe,0x03,0x1f]
# CHECK-NOFP: vcmp.s32 gt, q0, q1  @ encoding: [0x21,0xfe,0x03,0x1f]
vcmp.s32 gt, q0, q1

# CHECK: vcmp.s32 le, q5, q4  @ encoding: [0x2b,0xfe,0x89,0x1f]
# CHECK-NOFP: vcmp.s32 le, q5, q4  @ encoding: [0x2b,0xfe,0x89,0x1f]
vcmp.s32 le, q5, q4

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for signed integer comparison must be EQ, NE, LT, GT, LE or GE
vcmp.s32 ls, q2, q6

# CHECK: vcmp.u32 hi, q1, q4  @ encoding: [0x23,0xfe,0x89,0x0f]
# CHECK-NOFP: vcmp.u32 hi, q1, q4  @ encoding: [0x23,0xfe,0x89,0x0f]
vcmp.u32 hi, q1, q4

# CHECK: vcmp.u32 cs, q1, q4  @ encoding: [0x23,0xfe,0x09,0x0f]
# CHECK-NOFP: vcmp.u32 cs, q1, q4  @ encoding: [0x23,0xfe,0x09,0x0f]
vcmp.u32 cs, q1, q4

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for unsigned integer comparison must be EQ, NE, HS or HI
vcmp.u32 vs, q2, q6

# ERROR: [[@LINE+1]]:10: {{error|note}}: condition code for unsigned integer comparison must be EQ, NE, HS or HI
vcmp.u32 mi, q2, q6

# CHECK: vcmp.f16 gt, q4, zr  @ encoding: [0x39,0xfe,0x6f,0x1f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f16 gt, q4, zr  @ encoding: [0x39,0xfe,0x6f,0x1f]
vcmp.f16 gt, q4, zr

# CHECK: vcmp.f16 eq, q4, r12  @ encoding: [0x39,0xfe,0x4c,0x0f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f16 eq, q4, r12  @ encoding: [0x39,0xfe,0x4c,0x0f]
vcmp.f16 eq, q4, r12

# CHECK: vcmp.f32 ne, q3, r0  @ encoding: [0x37,0xee,0xc0,0x0f]
# ERROR-NOFP: [[@LINE+2]]:1: {{error|note}}: instruction requires: mve.fp
# CHECK-NOFP-NOT: vcmp.f32 ne, q3, r0  @ encoding: [0x37,0xee,0xc0,0x0f]
vcmp.f32 ne, q3, r0

# CHECK: vcmp.i8 eq, q1, r0  @ encoding: [0x03,0xfe,0x40,0x0f]
# CHECK-NOFP: vcmp.i8 eq, q1, r0  @ encoding: [0x03,0xfe,0x40,0x0f]
vcmp.i8 eq, q1, r0

# CHECK: vcmp.s8 le, q1, r0  @ encoding: [0x03,0xfe,0xe0,0x1f]
# CHECK-NOFP: vcmp.s8 le, q1, r0  @ encoding: [0x03,0xfe,0xe0,0x1f]
vcmp.s8 le, q1, r0

# CHECK: vcmp.u8 cs, q1, r0  @ encoding: [0x03,0xfe,0x60,0x0f]
# CHECK-NOFP: vcmp.u8 cs, q1, r0  @ encoding: [0x03,0xfe,0x60,0x0f]
vcmp.u8 cs, q1, r0

# CHECK: vcmp.i16 eq, q5, r10  @ encoding: [0x1b,0xfe,0x4a,0x0f]
# CHECK-NOFP: vcmp.i16 eq, q5, r10  @ encoding: [0x1b,0xfe,0x4a,0x0f]
vcmp.i16 eq, q5, r10

# CHECK: vcmp.i32 eq, q1, r4  @ encoding: [0x23,0xfe,0x44,0x0f]
# CHECK-NOFP: vcmp.i32 eq, q1, r4  @ encoding: [0x23,0xfe,0x44,0x0f]
vcmp.i32 eq, q1, r4

vpste
vcmpt.i8 eq, q0, r0
vcmpe.i16 ne, q0, r0
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vcmpt.i8 eq, q0, r0 @ encoding: [0x01,0xfe,0x40,0x0f]
# CHECK-NOFP: vcmpt.i8 eq, q0, r0 @ encoding: [0x01,0xfe,0x40,0x0f]
# CHECK: vcmpe.i16 ne, q0, r0 @ encoding: [0x11,0xfe,0xc0,0x0f]
# CHECK-NOFP: vcmpe.i16 ne, q0, r0 @ encoding: [0x11,0xfe,0xc0,0x0f]

# Ensure the scalar FP instructions VCMP and VCMPE are still correctly
# distinguished, in spite of VCMPE sometimes being a VPT-suffixed
# version of VCMP with identical encoding.
vcmp.f16  s0,s1
vcmpe.f16 s0,s1
# CHECK: vcmp.f16 s0, s1 @ encoding: [0xb4,0xee,0x60,0x09]
# CHECK: vcmpe.f16 s0, s1 @ encoding: [0xb4,0xee,0xe0,0x09]
# CHECK-NOFP-NOT: vcmp.f16 s0, s1 @ encoding: [0xb4,0xee,0x60,0x09]
# CHECK-NOFP-NOT: vcmpe.f16 s0, s1 @ encoding: [0xb4,0xee,0xe0,0x09]

itt eq
vcmpeq.f32 s0, s1
vcmpeeq.f32 s0, s1
# CHECK: itt eq @ encoding: [0x04,0xbf]
# CHECK: vcmpeq.f32 s0, s1 @ encoding: [0xb4,0xee,0x60,0x0a]
# CHECK-NOFP-NOT: vcmpeq.f32 s0, s1 @ encoding: [0xb4,0xee,0x60,0x0a]
# CHECK: vcmpeeq.f32 s0, s1 @ encoding: [0xb4,0xee,0xe0,0x0a]
# CHECK-NOFP-NOT: vcmpeeq.f32 s0, s1 @ encoding: [0xb4,0xee,0xe0,0x0a]
