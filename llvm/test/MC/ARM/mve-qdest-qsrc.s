# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s

# CHECK: vcvtb.f16.f32 q1, q4  @ encoding: [0x3f,0xee,0x09,0x2e]
# CHECK-NOFP-NOT: vcvtb.f16.f32 q1, q4  @ encoding: [0x3f,0xee,0x09,0x2e]
vcvtb.f16.f32 q1, q4

# CHECK: vcvtt.f32.f16 q0, q1  @ encoding: [0x3f,0xfe,0x03,0x1e]
# CHECK-NOFP-NOT: vcvtt.f32.f16 q0, q1  @ encoding: [0x3f,0xfe,0x03,0x1e]
vcvtt.f32.f16 q0, q1

# CHECK: vcvtt.f64.f16 d0, s0 @ encoding: [0xb2,0xee,0xc0,0x0b]
# CHECK-NOFP-NOT: vcvtt.f64.f16 d0, s0 @ encoding: [0xb2,0xee,0xc0,0x0b]
vcvtt.f64.f16 d0, s0

# CHECK: vcvtt.f16.f64 s1, d2 @ encoding: [0xf3,0xee,0xc2,0x0b]
# CHECK-NOFP-NOT: vcvtt.f16.f64 s1, d2 @ encoding: [0xf3,0xee,0xc2,0x0b]
vcvtt.f16.f64 s1, d2

# CHECK: vcvtt.f16.f32 q1, q4 @ encoding: [0x3f,0xee,0x09,0x3e]
# CHECK-NOFP-NOT: vcvtt.f16.f32 q1, q4 @ encoding: [0x3f,0xee,0x09,0x3e]
vcvtt.f16.f32 q1, q4

# CHECK: vqdmladhx.s8 q1, q6, q6  @ encoding: [0x0c,0xee,0x0c,0x3e]
# CHECK-NOFP: vqdmladhx.s8 q1, q6, q6  @ encoding: [0x0c,0xee,0x0c,0x3e]
vqdmladhx.s8 q1, q6, q6

# CHECK: vqdmladhx.s16 q0, q1, q4  @ encoding: [0x12,0xee,0x08,0x1e]
# CHECK-NOFP: vqdmladhx.s16 q0, q1, q4  @ encoding: [0x12,0xee,0x08,0x1e]
vqdmladhx.s16 q0, q1, q4

# CHECK: vqdmladhx.s32 q0, q3, q7  @ encoding: [0x26,0xee,0x0e,0x1e]
# CHECK-NOFP: vqdmladhx.s32 q0, q3, q7  @ encoding: [0x26,0xee,0x0e,0x1e]
vqdmladhx.s32 q0, q3, q7

# CHECK: vqdmladh.s8 q0, q1, q1  @ encoding: [0x02,0xee,0x02,0x0e]
# CHECK-NOFP: vqdmladh.s8 q0, q1, q1  @ encoding: [0x02,0xee,0x02,0x0e]
vqdmladh.s8 q0, q1, q1

# CHECK: vqdmladh.s16 q0, q2, q2  @ encoding: [0x14,0xee,0x04,0x0e]
# CHECK-NOFP: vqdmladh.s16 q0, q2, q2  @ encoding: [0x14,0xee,0x04,0x0e]
vqdmladh.s16 q0, q2, q2

# CHECK: vqdmladh.s32 q1, q5, q7  @ encoding: [0x2a,0xee,0x0e,0x2e]
# CHECK-NOFP: vqdmladh.s32 q1, q5, q7  @ encoding: [0x2a,0xee,0x0e,0x2e]
vqdmladh.s32 q1, q5, q7

# CHECK: vqrdmladhx.s8 q0, q7, q0  @ encoding: [0x0e,0xee,0x01,0x1e]
# CHECK-NOFP: vqrdmladhx.s8 q0, q7, q0  @ encoding: [0x0e,0xee,0x01,0x1e]
vqrdmladhx.s8 q0, q7, q0

# CHECK: vqrdmladhx.s16 q0, q0, q1  @ encoding: [0x10,0xee,0x03,0x1e]
# CHECK-NOFP: vqrdmladhx.s16 q0, q0, q1  @ encoding: [0x10,0xee,0x03,0x1e]
vqrdmladhx.s16 q0, q0, q1

# CHECK: vqrdmladhx.s32 q1, q0, q4  @ encoding: [0x20,0xee,0x09,0x3e]
# CHECK-NOFP: vqrdmladhx.s32 q1, q0, q4  @ encoding: [0x20,0xee,0x09,0x3e]
vqrdmladhx.s32 q1, q0, q4

# CHECK: vqrdmladhx.s32 q1, q1, q0  @ encoding: [0x22,0xee,0x01,0x3e]
# CHECK-NOFP: vqrdmladhx.s32 q1, q1, q0  @ encoding: [0x22,0xee,0x01,0x3e]
vqrdmladhx.s32 q1, q1, q0

# CHECK: vqrdmladhx.s32 q1, q0, q1  @ encoding: [0x20,0xee,0x03,0x3e]
# CHECK-NOFP: vqrdmladhx.s32 q1, q0, q1  @ encoding: [0x20,0xee,0x03,0x3e]
vqrdmladhx.s32 q1, q0, q1

# CHECK: vqrdmladh.s8 q0, q6, q2  @ encoding: [0x0c,0xee,0x05,0x0e]
# CHECK-NOFP: vqrdmladh.s8 q0, q6, q2  @ encoding: [0x0c,0xee,0x05,0x0e]
vqrdmladh.s8 q0, q6, q2

# CHECK: vqrdmladh.s16 q1, q5, q4  @ encoding: [0x1a,0xee,0x09,0x2e]
# CHECK-NOFP: vqrdmladh.s16 q1, q5, q4  @ encoding: [0x1a,0xee,0x09,0x2e]
vqrdmladh.s16 q1, q5, q4

# CHECK: vqrdmladh.s32 q0, q2, q2  @ encoding: [0x24,0xee,0x05,0x0e]
# CHECK-NOFP: vqrdmladh.s32 q0, q2, q2  @ encoding: [0x24,0xee,0x05,0x0e]
vqrdmladh.s32 q0, q2, q2

# CHECK: vqdmlsdhx.s8 q1, q4, q7  @ encoding: [0x08,0xfe,0x0e,0x3e]
# CHECK-NOFP: vqdmlsdhx.s8 q1, q4, q7  @ encoding: [0x08,0xfe,0x0e,0x3e]
vqdmlsdhx.s8 q1, q4, q7

# CHECK: vqdmlsdhx.s16 q0, q2, q5  @ encoding: [0x14,0xfe,0x0a,0x1e]
# CHECK-NOFP: vqdmlsdhx.s16 q0, q2, q5  @ encoding: [0x14,0xfe,0x0a,0x1e]
vqdmlsdhx.s16 q0, q2, q5

# CHECK: vqdmlsdhx.s32 q3, q4, q6  @ encoding: [0x28,0xfe,0x0c,0x7e]
# CHECK-NOFP: vqdmlsdhx.s32 q3, q4, q6  @ encoding: [0x28,0xfe,0x0c,0x7e]
vqdmlsdhx.s32 q3, q4, q6

# CHECK: vqdmlsdh.s8 q0, q3, q6  @ encoding: [0x06,0xfe,0x0c,0x0e]
# CHECK-NOFP: vqdmlsdh.s8 q0, q3, q6  @ encoding: [0x06,0xfe,0x0c,0x0e]
vqdmlsdh.s8 q0, q3, q6

# CHECK: vqdmlsdh.s16 q0, q4, q1  @ encoding: [0x18,0xfe,0x02,0x0e]
# CHECK-NOFP: vqdmlsdh.s16 q0, q4, q1  @ encoding: [0x18,0xfe,0x02,0x0e]
vqdmlsdh.s16 q0, q4, q1

# CHECK: vqdmlsdh.s32 q2, q5, q0  @ encoding: [0x2a,0xfe,0x00,0x4e]
# CHECK-NOFP: vqdmlsdh.s32 q2, q5, q0  @ encoding: [0x2a,0xfe,0x00,0x4e]
vqdmlsdh.s32 q2, q5, q0

# CHECK: vqrdmlsdhx.s8 q0, q3, q1  @ encoding: [0x06,0xfe,0x03,0x1e]
# CHECK-NOFP: vqrdmlsdhx.s8 q0, q3, q1  @ encoding: [0x06,0xfe,0x03,0x1e]
vqrdmlsdhx.s8 q0, q3, q1

# CHECK: vqrdmlsdhx.s16 q0, q1, q4  @ encoding: [0x12,0xfe,0x09,0x1e]
# CHECK-NOFP: vqrdmlsdhx.s16 q0, q1, q4  @ encoding: [0x12,0xfe,0x09,0x1e]
vqrdmlsdhx.s16 q0, q1, q4

# CHECK: vqrdmlsdhx.s32 q1, q6, q3  @ encoding: [0x2c,0xfe,0x07,0x3e]
# CHECK-NOFP: vqrdmlsdhx.s32 q1, q6, q3  @ encoding: [0x2c,0xfe,0x07,0x3e]
vqrdmlsdhx.s32 q1, q6, q3

# CHECK: vqrdmlsdh.s8 q3, q3, q0  @ encoding: [0x06,0xfe,0x01,0x6e]
# CHECK-NOFP: vqrdmlsdh.s8 q3, q3, q0  @ encoding: [0x06,0xfe,0x01,0x6e]
vqrdmlsdh.s8 q3, q3, q0

# CHECK: vqrdmlsdh.s16 q0, q7, q4  @ encoding: [0x1e,0xfe,0x09,0x0e]
# CHECK-NOFP: vqrdmlsdh.s16 q0, q7, q4  @ encoding: [0x1e,0xfe,0x09,0x0e]
vqrdmlsdh.s16 q0, q7, q4

# CHECK: vqrdmlsdh.s32 q0, q6, q7  @ encoding: [0x2c,0xfe,0x0f,0x0e]
# CHECK-NOFP: vqrdmlsdh.s32 q0, q6, q7  @ encoding: [0x2c,0xfe,0x0f,0x0e]
vqrdmlsdh.s32 q0, q6, q7

# CHECK: vqrdmlsdh.s32 q0, q0, q7  @ encoding: [0x20,0xfe,0x0f,0x0e]
# CHECK-NOFP: vqrdmlsdh.s32 q0, q0, q7  @ encoding: [0x20,0xfe,0x0f,0x0e]
vqrdmlsdh.s32 q0, q0, q7

# CHECK: vqrdmlsdh.s32 q0, q6, q0  @ encoding: [0x2c,0xfe,0x01,0x0e]
# CHECK-NOFP: vqrdmlsdh.s32 q0, q6, q0  @ encoding: [0x2c,0xfe,0x01,0x0e]
vqrdmlsdh.s32 q0, q6, q0

# CHECK: vcmul.f16 q0, q1, q2, #90 @ encoding: [0x32,0xee,0x05,0x0e]
# CHECK-NOFP-NOT: vcmul.f16 q0, q1, q2, #90 @ encoding: [0x32,0xee,0x05,0x0e]
vcmul.f16 q0, q1, q2, #90

# CHECK: vcmul.f16 q6, q2, q5, #0  @ encoding: [0x34,0xee,0x0a,0xce]
# CHECK-NOFP-NOT: vcmul.f16 q6, q2, q5, #0  @ encoding: [0x34,0xee,0x0a,0xce]
vcmul.f16 q6, q2, q5, #0

# CHECK: vcmul.f16 q1, q0, q5, #90  @ encoding: [0x30,0xee,0x0b,0x2e]
# CHECK-NOFP-NOT: vcmul.f16 q1, q0, q5, #90  @ encoding: [0x30,0xee,0x0b,0x2e]
vcmul.f16 q1, q0, q5, #90

# CHECK: vcmul.f16 q1, q0, q5, #180  @ encoding: [0x30,0xee,0x0a,0x3e]
# CHECK-NOFP-NOT: vcmul.f16 q1, q0, q5, #180  @ encoding: [0x30,0xee,0x0a,0x3e]
vcmul.f16 q1, q0, q5, #180

# CHECK: vcmul.f16 q1, q0, q5, #270  @ encoding: [0x30,0xee,0x0b,0x3e]
# CHECK-NOFP-NOT: vcmul.f16 q1, q0, q5, #270  @ encoding: [0x30,0xee,0x0b,0x3e]
vcmul.f16 q1, q0, q5, #270

# CHECK: vcmul.f16 q1, q0, q1, #270  @ encoding: [0x30,0xee,0x03,0x3e]
# CHECK-NOFP-NOT: vcmul.f16 q1, q0, q1, #270  @ encoding: [0x30,0xee,0x03,0x3e]
vcmul.f16 q1, q0, q1, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 0, 90, 180 or 270
vcmul.f16 q1, q0, q5, #300

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qn register can't be identical
vcmul.f32 q1, q1, q5, #0

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qm register can't be identical
vcmul.f32 q1, q5, q1, #0

# CHECK: vcmul.f32 q1, q7, q5, #0  @ encoding: [0x3e,0xfe,0x0a,0x2e]
# CHECK-NOFP-NOT: vcmul.f32 q1, q7, q5, #0  @ encoding: [0x3e,0xfe,0x0a,0x2e]
vcmul.f32 q1, q7, q5, #0

# CHECK: vcmul.f32 q3, q4, q2, #90  @ encoding: [0x38,0xfe,0x05,0x6e]
# CHECK-NOFP-NOT: vcmul.f32 q3, q4, q2, #90  @ encoding: [0x38,0xfe,0x05,0x6e]
vcmul.f32 q3, q4, q2, #90

# CHECK: vcmul.f32 q5, q1, q3, #180  @ encoding: [0x32,0xfe,0x06,0xbe]
# CHECK-NOFP-NOT: vcmul.f32 q5, q1, q3, #180  @ encoding: [0x32,0xfe,0x06,0xbe]
vcmul.f32 q5, q1, q3, #180

# CHECK: vcmul.f32 q0, q7, q4, #270  @ encoding: [0x3e,0xfe,0x09,0x1e]
# CHECK-NOFP-NOT: vcmul.f32 q0, q7, q4, #270  @ encoding: [0x3e,0xfe,0x09,0x1e]
vcmul.f32 q0, q7, q4, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 0, 90, 180 or 270
vcmul.f32 q1, q0, q5, #300

# CHECK: vmullb.s8 q2, q6, q0  @ encoding: [0x0d,0xee,0x00,0x4e]
# CHECK-NOFP: vmullb.s8 q2, q6, q0  @ encoding: [0x0d,0xee,0x00,0x4e]
vmullb.s8 q2, q6, q0

# CHECK: vmullb.s16 q3, q4, q3  @ encoding: [0x19,0xee,0x06,0x6e]
# CHECK-NOFP: vmullb.s16 q3, q4, q3  @ encoding: [0x19,0xee,0x06,0x6e]
vmullb.s16 q3, q4, q3

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qm register can't be identical
vmullb.s32 q3, q4, q3

# CHECK: vmullb.s32 q3, q5, q6  @ encoding: [0x2b,0xee,0x0c,0x6e]
# CHECK-NOFP: vmullb.s32 q3, q5, q6  @ encoding: [0x2b,0xee,0x0c,0x6e]
vmullb.s32 q3, q5, q6

# CHECK: vmullt.s8 q0, q6, q2  @ encoding: [0x0d,0xee,0x04,0x1e]
# CHECK-NOFP: vmullt.s8 q0, q6, q2  @ encoding: [0x0d,0xee,0x04,0x1e]
vmullt.s8 q0, q6, q2

# CHECK: vmullt.s16 q0, q0, q2  @ encoding: [0x11,0xee,0x04,0x1e]
# CHECK-NOFP: vmullt.s16 q0, q0, q2  @ encoding: [0x11,0xee,0x04,0x1e]
vmullt.s16 q0, q0, q2

# CHECK: vmullt.s32 q2, q4, q4  @ encoding: [0x29,0xee,0x08,0x5e]
# CHECK-NOFP: vmullt.s32 q2, q4, q4  @ encoding: [0x29,0xee,0x08,0x5e]
vmullt.s32 q2, q4, q4

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qn register can't be identical
vmullt.s32 q4, q4, q2

# CHECK: vmullb.p8 q2, q3, q7  @ encoding: [0x37,0xee,0x0e,0x4e]
# CHECK-NOFP: vmullb.p8 q2, q3, q7  @ encoding: [0x37,0xee,0x0e,0x4e]
vmullb.p8 q2, q3, q7

# CHECK: vmullb.p16 q0, q1, q3  @ encoding: [0x33,0xfe,0x06,0x0e]
# CHECK-NOFP: vmullb.p16 q0, q1, q3  @ encoding: [0x33,0xfe,0x06,0x0e]
vmullb.p16 q0, q1, q3

# CHECK: vmullt.p8 q1, q1, q7  @ encoding: [0x33,0xee,0x0e,0x3e]
# CHECK-NOFP: vmullt.p8 q1, q1, q7  @ encoding: [0x33,0xee,0x0e,0x3e]
vmullt.p8 q1, q1, q7

# CHECK: vmullt.p16 q0, q7, q7  @ encoding: [0x3f,0xfe,0x0e,0x1e]
# CHECK-NOFP: vmullt.p16 q0, q7, q7  @ encoding: [0x3f,0xfe,0x0e,0x1e]
vmullt.p16 q0, q7, q7

# CHECK: vmulh.s8 q0, q4, q5  @ encoding: [0x09,0xee,0x0b,0x0e]
# CHECK-NOFP: vmulh.s8 q0, q4, q5  @ encoding: [0x09,0xee,0x0b,0x0e]
vmulh.s8 q0, q4, q5

# CHECK: vmulh.s16 q0, q7, q4  @ encoding: [0x1f,0xee,0x09,0x0e]
# CHECK-NOFP: vmulh.s16 q0, q7, q4  @ encoding: [0x1f,0xee,0x09,0x0e]
vmulh.s16 q0, q7, q4

# CHECK: vmulh.s32 q0, q7, q4  @ encoding: [0x2f,0xee,0x09,0x0e]
# CHECK-NOFP: vmulh.s32 q0, q7, q4  @ encoding: [0x2f,0xee,0x09,0x0e]
vmulh.s32 q0, q7, q4

# CHECK: vmulh.u8 q3, q5, q2  @ encoding: [0x0b,0xfe,0x05,0x6e]
# CHECK-NOFP: vmulh.u8 q3, q5, q2  @ encoding: [0x0b,0xfe,0x05,0x6e]
vmulh.u8 q3, q5, q2

# CHECK: vmulh.u16 q2, q7, q4  @ encoding: [0x1f,0xfe,0x09,0x4e]
# CHECK-NOFP: vmulh.u16 q2, q7, q4  @ encoding: [0x1f,0xfe,0x09,0x4e]
vmulh.u16 q2, q7, q4

# CHECK: vmulh.u32 q1, q3, q2  @ encoding: [0x27,0xfe,0x05,0x2e]
# CHECK-NOFP: vmulh.u32 q1, q3, q2  @ encoding: [0x27,0xfe,0x05,0x2e]
vmulh.u32 q1, q3, q2

# CHECK: vrmulh.s8 q1, q1, q2  @ encoding: [0x03,0xee,0x05,0x3e]
# CHECK-NOFP: vrmulh.s8 q1, q1, q2  @ encoding: [0x03,0xee,0x05,0x3e]
vrmulh.s8 q1, q1, q2

# CHECK: vrmulh.s16 q1, q1, q2  @ encoding: [0x13,0xee,0x05,0x3e]
# CHECK-NOFP: vrmulh.s16 q1, q1, q2  @ encoding: [0x13,0xee,0x05,0x3e]
vrmulh.s16 q1, q1, q2

# CHECK: vrmulh.s32 q3, q1, q0  @ encoding: [0x23,0xee,0x01,0x7e]
# CHECK-NOFP: vrmulh.s32 q3, q1, q0  @ encoding: [0x23,0xee,0x01,0x7e]
vrmulh.s32 q3, q1, q0

# CHECK: vrmulh.u8 q1, q6, q0  @ encoding: [0x0d,0xfe,0x01,0x3e]
# CHECK-NOFP: vrmulh.u8 q1, q6, q0  @ encoding: [0x0d,0xfe,0x01,0x3e]
vrmulh.u8 q1, q6, q0

# CHECK: vrmulh.u16 q4, q3, q6  @ encoding: [0x17,0xfe,0x0d,0x9e]
# CHECK-NOFP: vrmulh.u16 q4, q3, q6  @ encoding: [0x17,0xfe,0x0d,0x9e]
vrmulh.u16 q4, q3, q6

# CHECK: vrmulh.u32 q1, q2, q2  @ encoding: [0x25,0xfe,0x05,0x3e]
# CHECK-NOFP: vrmulh.u32 q1, q2, q2  @ encoding: [0x25,0xfe,0x05,0x3e]
vrmulh.u32 q1, q2, q2

# CHECK: vqmovnb.s16 q0, q1  @ encoding: [0x33,0xee,0x03,0x0e]
# CHECK-NOFP: vqmovnb.s16 q0, q1  @ encoding: [0x33,0xee,0x03,0x0e]
vqmovnb.s16 q0, q1

# CHECK: vqmovnt.s16 q2, q0  @ encoding: [0x33,0xee,0x01,0x5e]
# CHECK-NOFP: vqmovnt.s16 q2, q0  @ encoding: [0x33,0xee,0x01,0x5e]
vqmovnt.s16 q2, q0

# CHECK: vqmovnb.s32 q0, q5  @ encoding: [0x37,0xee,0x0b,0x0e]
# CHECK-NOFP: vqmovnb.s32 q0, q5  @ encoding: [0x37,0xee,0x0b,0x0e]
vqmovnb.s32 q0, q5

# CHECK: vqmovnt.s32 q0, q1  @ encoding: [0x37,0xee,0x03,0x1e]
# CHECK-NOFP: vqmovnt.s32 q0, q1  @ encoding: [0x37,0xee,0x03,0x1e]
vqmovnt.s32 q0, q1

# CHECK: vqmovnb.u16 q0, q4  @ encoding: [0x33,0xfe,0x09,0x0e]
# CHECK-NOFP: vqmovnb.u16 q0, q4  @ encoding: [0x33,0xfe,0x09,0x0e]
vqmovnb.u16 q0, q4

# CHECK: vqmovnt.u16 q0, q7  @ encoding: [0x33,0xfe,0x0f,0x1e]
# CHECK-NOFP: vqmovnt.u16 q0, q7  @ encoding: [0x33,0xfe,0x0f,0x1e]
vqmovnt.u16 q0, q7

# CHECK: vqmovnb.u32 q0, q4  @ encoding: [0x37,0xfe,0x09,0x0e]
# CHECK-NOFP: vqmovnb.u32 q0, q4  @ encoding: [0x37,0xfe,0x09,0x0e]
vqmovnb.u32 q0, q4

# CHECK: vqmovnt.u32 q0, q2  @ encoding: [0x37,0xfe,0x05,0x1e]
# CHECK-NOFP: vqmovnt.u32 q0, q2  @ encoding: [0x37,0xfe,0x05,0x1e]
vqmovnt.u32 q0, q2

# CHECK: vcvtb.f16.f32 q1, q4  @ encoding: [0x3f,0xee,0x09,0x2e]
# CHECK-NOFP-NOT: vcvtb.f16.f32 q1, q4  @ encoding: [0x3f,0xee,0x09,0x2e]
vcvtb.f16.f32 q1, q4

# CHECK: vcvtt.f16.f32 q1, q4  @ encoding: [0x3f,0xee,0x09,0x3e]
# CHECK-NOFP-NOT: vcvtt.f16.f32 q1, q4  @ encoding: [0x3f,0xee,0x09,0x3e]
vcvtt.f16.f32 q1, q4

# CHECK: vcvtb.f32.f16 q0, q3  @ encoding: [0x3f,0xfe,0x07,0x0e]
# CHECK-NOFP-NOT: vcvtb.f32.f16 q0, q3  @ encoding: [0x3f,0xfe,0x07,0x0e]
vcvtb.f32.f16 q0, q3

# CHECK: vcvtt.f32.f16 q0, q1  @ encoding: [0x3f,0xfe,0x03,0x1e]
# CHECK-NOFP-NOT: vcvtt.f32.f16 q0, q1  @ encoding: [0x3f,0xfe,0x03,0x1e]
vcvtt.f32.f16 q0, q1

# CHECK: vqmovunb.s16 q0, q3  @ encoding: [0x31,0xee,0x87,0x0e]
# CHECK-NOFP: vqmovunb.s16 q0, q3  @ encoding: [0x31,0xee,0x87,0x0e]
vqmovunb.s16 q0, q3

# CHECK: vqmovunt.s16 q4, q1  @ encoding: [0x31,0xee,0x83,0x9e]
# CHECK-NOFP: vqmovunt.s16 q4, q1  @ encoding: [0x31,0xee,0x83,0x9e]
vqmovunt.s16 q4, q1

# CHECK: vqmovunb.s32 q1, q7  @ encoding: [0x35,0xee,0x8f,0x2e]
# CHECK-NOFP: vqmovunb.s32 q1, q7  @ encoding: [0x35,0xee,0x8f,0x2e]
vqmovunb.s32 q1, q7

# CHECK: vqmovunt.s32 q0, q2  @ encoding: [0x35,0xee,0x85,0x1e]
# CHECK-NOFP: vqmovunt.s32 q0, q2  @ encoding: [0x35,0xee,0x85,0x1e]
vqmovunt.s32 q0, q2

# CHECK: vmovnb.i16 q1, q5  @ encoding: [0x31,0xfe,0x8b,0x2e]
# CHECK-NOFP: vmovnb.i16 q1, q5  @ encoding: [0x31,0xfe,0x8b,0x2e]
vmovnb.i16 q1, q5

# CHECK: vmovnt.i16 q0, q0  @ encoding: [0x31,0xfe,0x81,0x1e]
# CHECK-NOFP: vmovnt.i16 q0, q0  @ encoding: [0x31,0xfe,0x81,0x1e]
vmovnt.i16 q0, q0

# CHECK: vmovnb.i32 q1, q0  @ encoding: [0x35,0xfe,0x81,0x2e]
# CHECK-NOFP: vmovnb.i32 q1, q0  @ encoding: [0x35,0xfe,0x81,0x2e]
vmovnb.i32 q1, q0

# CHECK: vmovnt.i32 q3, q3  @ encoding: [0x35,0xfe,0x87,0x7e]
# CHECK-NOFP: vmovnt.i32 q3, q3  @ encoding: [0x35,0xfe,0x87,0x7e]
vmovnt.i32 q3, q3

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 90 or 270
vhcadd.s8 q3, q7, q5, #0

# CHECK: vhcadd.s8 q3, q7, q5, #90  @ encoding: [0x0e,0xee,0x0a,0x6f]
# CHECK-NOFP: vhcadd.s8 q3, q7, q5, #90  @ encoding: [0x0e,0xee,0x0a,0x6f]
vhcadd.s8 q3, q7, q5, #90

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 90 or 270
vhcadd.s8 q3, q7, q5, #0

# CHECK: vhcadd.s16 q0, q0, q6, #90  @ encoding: [0x10,0xee,0x0c,0x0f]
# CHECK-NOFP: vhcadd.s16 q0, q0, q6, #90  @ encoding: [0x10,0xee,0x0c,0x0f]
vhcadd.s16 q0, q0, q6, #90

# CHECK: vhcadd.s16 q0, q0, q6, #90  @ encoding: [0x10,0xee,0x0c,0x0f]
# CHECK-NOFP: vhcadd.s16 q0, q0, q6, #90  @ encoding: [0x10,0xee,0x0c,0x0f]
vhcadd.s16 q0, q0, q6, #90

# CHECK: vhcadd.s16 q3, q1, q0, #270  @ encoding: [0x12,0xee,0x00,0x7f]
# CHECK-NOFP: vhcadd.s16 q3, q1, q0, #270  @ encoding: [0x12,0xee,0x00,0x7f]
vhcadd.s16 q3, q1, q0, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 90 or 270
vhcadd.s32 q3, q4, q5, #0

# CHECK: vhcadd.s32 q3, q4, q5, #90  @ encoding: [0x28,0xee,0x0a,0x6f]
# CHECK-NOFP: vhcadd.s32 q3, q4, q5, #90  @ encoding: [0x28,0xee,0x0a,0x6f]
vhcadd.s32 q3, q4, q5, #90

# CHECK: vhcadd.s32 q6, q7, q2, #270  @ encoding: [0x2e,0xee,0x04,0xdf]
# CHECK-NOFP: vhcadd.s32 q6, q7, q2, #270  @ encoding: [0x2e,0xee,0x04,0xdf]
vhcadd.s32 q6, q7, q2, #270

# CHECK: vadc.i32 q1, q0, q2  @ encoding: [0x30,0xee,0x04,0x2f]
# CHECK-NOFP: vadc.i32 q1, q0, q2  @ encoding: [0x30,0xee,0x04,0x2f]
vadc.i32 q1, q0, q2

# CHECK: vadci.i32 q0, q1, q1  @ encoding: [0x32,0xee,0x02,0x1f]
# CHECK-NOFP: vadci.i32 q0, q1, q1  @ encoding: [0x32,0xee,0x02,0x1f]
vadci.i32 q0, q1, q1

# CHECK: vcadd.i8 q1, q0, q2, #90  @ encoding: [0x00,0xfe,0x04,0x2f]
# CHECK-NOFP: vcadd.i8 q1, q0, q2, #90  @ encoding: [0x00,0xfe,0x04,0x2f]
vcadd.i8 q1, q0, q2, #90

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 90 or 270
vcadd.i8 q1, q0, q2, #0

# CHECK: vcadd.i16 q0, q2, q3, #90  @ encoding: [0x14,0xfe,0x06,0x0f]
# CHECK-NOFP: vcadd.i16 q0, q2, q3, #90  @ encoding: [0x14,0xfe,0x06,0x0f]
vcadd.i16 q0, q2, q3, #90

# CHECK: vcadd.i16 q0, q5, q5, #270  @ encoding: [0x1a,0xfe,0x0a,0x1f]
# CHECK-NOFP: vcadd.i16 q0, q5, q5, #270  @ encoding: [0x1a,0xfe,0x0a,0x1f]
vcadd.i16 q0, q5, q5, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 90 or 270
vcadd.i16 q1, q0, q2, #0

# CHECK: vcadd.i32 q4, q2, q5, #90  @ encoding: [0x24,0xfe,0x0a,0x8f]
# CHECK-NOFP: vcadd.i32 q4, q2, q5, #90  @ encoding: [0x24,0xfe,0x0a,0x8f]
vcadd.i32 q4, q2, q5, #90

# CHECK: vcadd.i32 q5, q5, q0, #270  @ encoding: [0x2a,0xfe,0x00,0xbf]
# CHECK-NOFP: vcadd.i32 q5, q5, q0, #270  @ encoding: [0x2a,0xfe,0x00,0xbf]
vcadd.i32 q5, q5, q0, #270

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: complex rotation must be 90 or 270
vcadd.i32 q4, q2, q5, #0

# CHECK: vsbc.i32 q3, q1, q1  @ encoding: [0x32,0xfe,0x02,0x6f]
# CHECK-NOFP: vsbc.i32 q3, q1, q1  @ encoding: [0x32,0xfe,0x02,0x6f]
vsbc.i32 q3, q1, q1

# CHECK: vsbci.i32 q2, q6, q2  @ encoding: [0x3c,0xfe,0x04,0x5f]
# CHECK-NOFP: vsbci.i32 q2, q6, q2  @ encoding: [0x3c,0xfe,0x04,0x5f]
vsbci.i32 q2, q6, q2

# CHECK: vqdmullb.s16 q0, q4, q5  @ encoding: [0x38,0xee,0x0b,0x0f]
# CHECK-NOFP: vqdmullb.s16 q0, q4, q5  @ encoding: [0x38,0xee,0x0b,0x0f]
vqdmullb.s16 q0, q4, q5

# CHECK: vqdmullt.s16 q0, q6, q5  @ encoding: [0x3c,0xee,0x0b,0x1f]
# CHECK-NOFP: vqdmullt.s16 q0, q6, q5  @ encoding: [0x3c,0xee,0x0b,0x1f]
vqdmullt.s16 q0, q6, q5

# CHECK: vqdmullb.s32 q0, q3, q7  @ encoding: [0x36,0xfe,0x0f,0x0f]
# CHECK-NOFP: vqdmullb.s32 q0, q3, q7  @ encoding: [0x36,0xfe,0x0f,0x0f]
vqdmullb.s32 q0, q3, q7

# CHECK: vqdmullt.s32 q0, q7, q5  @ encoding: [0x3e,0xfe,0x0b,0x1f]
# CHECK-NOFP: vqdmullt.s32 q0, q7, q5  @ encoding: [0x3e,0xfe,0x0b,0x1f]
vqdmullt.s32 q0, q7, q5

# CHECK: vqdmullb.s16 q0, q1, q0  @ encoding: [0x32,0xee,0x01,0x0f]
# CHECK-NOFP: vqdmullb.s16 q0, q1, q0  @ encoding: [0x32,0xee,0x01,0x0f]
vqdmullb.s16 q0, q1, q0

# CHECK: vqdmullt.s16 q0, q0, q5  @ encoding: [0x30,0xee,0x0b,0x1f]
# CHECK-NOFP: vqdmullt.s16 q0, q0, q5  @ encoding: [0x30,0xee,0x0b,0x1f]
vqdmullt.s16 q0, q0, q5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qm register can't be identical
vqdmullb.s32 q0, q1, q0
# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qn register can't be identical
vqdmullb.s32 q0, q0, q1

vqdmullt.s16 q0, q1, q2
# CHECK: vqdmullt.s16 q0, q1, q2 @ encoding: [0x32,0xee,0x05,0x1f]
# CHECK-NOFP: vqdmullt.s16 q0, q1, q2 @ encoding: [0x32,0xee,0x05,0x1f]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qn register can't be identical
vqdmullb.s32 q0, q0, r0
# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qn register can't be identical
vqdmullt.s32 q0, q0, r0

vqdmullb.s16 q0, q0, r0
# CHECK: vqdmullb.s16 q0, q0, r0 @ encoding: [0x30,0xee,0x60,0x0f]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qm register can't be identical
vcadd.s32 q0, q1, q0, #270
# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qm register can't be identical
vcadd.f32 q0, q1, q0, #270
# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qm register can't be identical
vhcadd.s32 q0, q1, q0, #270

vcadd.s32 q0, q0, q1, #270
# CHECK: vcadd.i32 q0, q0, q1, #270 @ encoding: [0x20,0xfe,0x02,0x1f]
vcadd.f32 q0, q0, q1, #270
# CHECK: vcadd.f32 q0, q0, q1, #270 @ encoding: [0x90,0xfd,0x42,0x08]
vhcadd.s32 q0, q0, q1, #270
# CHECK: vhcadd.s32 q0, q0, q1, #270 @ encoding: [0x20,0xee,0x02,0x1f]
vhcadd.s16 q0, q0, q1, #270
# CHECK: vhcadd.s16 q0, q0, q1, #270 @ encoding: [0x10,0xee,0x02,0x1f]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qn register can't be identical
vrev64.8 q0, q0
# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qn register can't be identical
vrev64.16 q0, q0
# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Qd register and Qn register can't be identical
vrev64.32 q0, q0

vrev32.8 q0, q0
# CHECK: vrev32.8 q0, q0 @ encoding: [0xb0,0xff,0xc0,0x00]

vpste
vqdmulltt.s32 q0, q1, q2
vqdmullbe.s16 q0, q1, q2
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vqdmulltt.s32 q0, q1, q2 @ encoding: [0x32,0xfe,0x05,0x1f]
# CHECK-NOFP: vqdmulltt.s32 q0, q1, q2 @ encoding: [0x32,0xfe,0x05,0x1f]
# CHECK: vqdmullbe.s16 q0, q1, q2 @ encoding: [0x32,0xee,0x05,0x0f]
# CHECK-NOFP: vqdmullbe.s16 q0, q1, q2 @ encoding: [0x32,0xee,0x05,0x0f]

vpste
vmulltt.p8 q0, q1, q2
vmullbe.p16 q0, q1, q2
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vmulltt.p8 q0, q1, q2 @ encoding: [0x33,0xee,0x04,0x1e]
# CHECK-NOFP: vmulltt.p8 q0, q1, q2 @ encoding: [0x33,0xee,0x04,0x1e]
# CHECK: vmullbe.p16 q0, q1, q2 @ encoding: [0x33,0xfe,0x04,0x0e]
# CHECK-NOFP: vmullbe.p16 q0, q1, q2 @ encoding: [0x33,0xfe,0x04,0x0e]

# ----------------------------------------------------------------------
# The following tests have to go last because of the NOFP-NOT checks inside the
# VPT block.

vpste
vcmult.f16 q0, q1, q2, #180
vcmule.f16 q0, q1, q2, #180
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vcmult.f16 q0, q1, q2, #180 @ encoding: [0x32,0xee,0x04,0x1e]
# CHECK-NOFP-NOT: vcmult.f16 q0, q1, q2, #180 @ encoding: [0x32,0xee,0x04,0x1e]
# CHECK: vcmule.f16 q0, q1, q2, #180 @ encoding: [0x32,0xee,0x04,0x1e]
# CHECK-NOFP-NOT: vcmule.f16 q0, q1, q2, #180 @ encoding: [0x32,0xee,0x04,0x1e]

vpstet
vcvtbt.f16.f32 q0, q1
vcvtne.s16.f16 q0, q1
vcvtmt.s16.f16 q0, q1
# CHECK: vpstet @ encoding: [0x71,0xfe,0x4d,0xcf]
# CHECK: vcvtbt.f16.f32 q0, q1 @ encoding: [0x3f,0xee,0x03,0x0e]
# CHECK-NOFP-NOT: vcvtbt.f16.f32 q0, q1 @ encoding: [0x3f,0xee,0x03,0x0e]
# CHECK: vcvtne.s16.f16 q0, q1 @ encoding: [0xb7,0xff,0x42,0x01]
# CHECK-NOFP-NOT: vcvtne.s16.f16 q0, q1 @ encoding: [0xb7,0xff,0x42,0x01]
# CHECK: vcvtmt.s16.f16 q0, q1 @ encoding: [0xb7,0xff,0x42,0x03
# CHECK-NOFP-NOT: vcvtmt.s16.f16 q0, q1 @ encoding: [0xb7,0xff,0x42,0x03

vpte.f32 lt, q3, r1
vcvttt.f16.f32 q2, q0
vcvtte.f32.f16 q1, q0
# CHECK: vpte.f32 lt, q3, r1      @ encoding: [0x77,0xee,0xc1,0x9f]
# CHECK-NOFP-NOT: vpte.f32 lt, q3, r1      @ encoding: [0x77,0xee,0xe1,0x8f]
# CHECK: vcvttt.f16.f32 q2, q0          @ encoding: [0x3f,0xee,0x01,0x5e]
# CHECK-NOFP-NOT: vcvttt.f16.f32 q2, q0          @ encoding: [0x3f,0xee,0x01,0x5e]
# CHECK: vcvtte.f32.f16 q1, q0          @ encoding: [0x3f,0xfe,0x01,0x3e]

vpte.f32 lt, q3, r1
vcvtbt.f16.f32 q2, q0
vcvtbe.f32.f16 q1, q0
# CHECK: vpte.f32 lt, q3, r1      @ encoding: [0x77,0xee,0xc1,0x9f]
# CHECK-NOFP-NOT: vpte.f32 lt, q3, r1      @ encoding: [0x77,0xee,0xe1,0x8f]
# CHECK: vcvtbt.f16.f32 q2, q0          @ encoding: [0x3f,0xee,0x01,0x4e]
# CHECK-NOFP-NOT: vcvtbt.f16.f32 q2, q0          @ encoding: [0x3f,0xee,0x01,0x4e]
# CHECK: vcvtbe.f32.f16 q1, q0          @ encoding: [0x3f,0xfe,0x01,0x2e]
# CHECK-NOFP-NOT: vcvtbe.f32.f16 q1, q0          @ encoding: [0x3f,0xfe,0x01,0x2e]

ite eq
vcvtteq.f16.f32 s0, s1
vcvttne.f16.f32 s0, s1
# CHECK: ite eq                      @ encoding: [0x0c,0xbf]
# CHECK: vcvtteq.f16.f32 s0, s1          @ encoding: [0xb3,0xee,0xe0,0x0a]
# CHECK-NOFP-NOT: vcvtteq.f16.f32 s0, s1          @ encoding: [0xb3,0xee,0xe0,0x0a]
# CHECK: vcvttne.f16.f32 s0, s1          @ encoding: [0xb3,0xee,0xe0,0x0a]
# CHECK-NOFP-NOT: vcvttne.f16.f32 s0, s1          @ encoding: [0xb3,0xee,0xe0,0x0a]
