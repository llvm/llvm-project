# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s

# CHECK: vorr.i16 q0, #0x12  @ encoding: [0x81,0xef,0x52,0x09]
# CHECK-NOFP: vorr.i16 q0, #0x12  @ encoding: [0x81,0xef,0x52,0x09]
vorr.i16 q0, #0x12

# CHECK: vorr.i32 q0, #0x1200  @ encoding: [0x81,0xef,0x52,0x03]
# CHECK-NOFP: vorr.i32 q0, #0x1200  @ encoding: [0x81,0xef,0x52,0x03]
vorr.i32 q0, #0x1200

# CHECK: vorr.i16 q0, #0xed  @ encoding: [0x86,0xff,0x5d,0x09]
# CHECK-NOFP: vorr.i16 q0, #0xed  @ encoding: [0x86,0xff,0x5d,0x09]
vorn.i16 q0, #0xff12

# CHECK: vorr.i32 q0, #0xed00  @ encoding: [0x86,0xff,0x5d,0x03]
# CHECK-NOFP: vorr.i32 q0, #0xed00  @ encoding: [0x86,0xff,0x5d,0x03]
vorn.i32 q0, #0xffff12ff

# CHECK: vorr.i32 q0, #0xed0000  @ encoding: [0x86,0xff,0x5d,0x05]
# CHECK-NOFP: vorr.i32 q0, #0xed0000  @ encoding: [0x86,0xff,0x5d,0x05]
vorn.i32 q0, #0xff12ffff

# CHECK: vorr.i32 q0, #0xed000000  @ encoding: [0x86,0xff,0x5d,0x07]
# CHECK-NOFP: vorr.i32 q0, #0xed000000  @ encoding: [0x86,0xff,0x5d,0x07]
vorn.i32 q0, #0x12ffffff

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vorn.i16 q0, #0xed00

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vorn.i16 q0, #0x00ed

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vorn.i32 q0, #0xed000000

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vorn.i32 q0, #0x00ed0000

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vorn.i32 q0, #0x0000ed00

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vorn.i32 q0, #0x000000ed

# CHECK: vbic.i16 q0, #0x22  @ encoding: [0x82,0xef,0x72,0x09]
# CHECK-NOFP: vbic.i16 q0, #0x22  @ encoding: [0x82,0xef,0x72,0x09]
vbic.i16 q0, #0x22

# CHECK: vbic.i32 q0, #0x1100  @ encoding: [0x81,0xef,0x71,0x03]
# CHECK-NOFP: vbic.i32 q0, #0x1100  @ encoding: [0x81,0xef,0x71,0x03]
vbic.i32 q0, #0x1100

# CHECK: vbic.i16 q0, #0xdd  @ encoding: [0x85,0xff,0x7d,0x09]
# CHECK-NOFP: vbic.i16 q0, #0xdd  @ encoding: [0x85,0xff,0x7d,0x09]
vand.i16 q0, #0xff22

# CHECK: vbic.i16 q0, #0xdd00  @ encoding: [0x85,0xff,0x7d,0x0b]
# CHECK-NOFP: vbic.i16 q0, #0xdd00  @ encoding: [0x85,0xff,0x7d,0x0b]
vand.i16 q0, #0x22ff

# CHECK: vbic.i32 q0, #0xee  @ encoding: [0x86,0xff,0x7e,0x01]
# CHECK-NOFP: vbic.i32 q0, #0xee  @ encoding: [0x86,0xff,0x7e,0x01]
vand.i32 q0, #0xffffff11

# CHECK: vbic.i32 q0, #0xee00  @ encoding: [0x86,0xff,0x7e,0x03]
# CHECK-NOFP: vbic.i32 q0, #0xee00  @ encoding: [0x86,0xff,0x7e,0x03]
vand.i32 q0, #0xffff11ff

# CHECK: vbic.i32 q0, #0xee0000  @ encoding: [0x86,0xff,0x7e,0x05]
# CHECK-NOFP: vbic.i32 q0, #0xee0000  @ encoding: [0x86,0xff,0x7e,0x05]
vand.i32 q0, #0xff11ffff

# CHECK: vbic.i32 q0, #0xee000000  @ encoding: [0x86,0xff,0x7e,0x07]
# CHECK-NOFP: vbic.i32 q0, #0xee000000  @ encoding: [0x86,0xff,0x7e,0x07]
vand.i32 q0, #0x11ffffff

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vand.i16 q0, #0xed00

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vand.i16 q0, #0x00ed

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vand.i32 q0, #0xed000000

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vand.i32 q0, #0x00ed0000

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vand.i32 q0, #0x0000ed00

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vand.i32 q0, #0x000000ed

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.s8 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.s16 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.s32 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.u8 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.u16 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.u32 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.i8 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.i16 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.i32 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.f16 q0, q1, q7

# CHECK: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
# CHECK-NOFP: vbic q0, q1, q7  @ encoding: [0x12,0xef,0x5e,0x01]
vbic.f32 q0, q1, q7

# CHECK: vrev64.8 q0, q4  @ encoding: [0xb0,0xff,0x48,0x00]
# CHECK-NOFP: vrev64.8 q0, q4  @ encoding: [0xb0,0xff,0x48,0x00]
vrev64.8 q0, q4

# CHECK: vrev64.16 q1, q3  @ encoding: [0xb4,0xff,0x46,0x20]
# CHECK-NOFP: vrev64.16 q1, q3  @ encoding: [0xb4,0xff,0x46,0x20]
vrev64.16 q1, q3

# CHECK: vrev64.32 q0, q2  @ encoding: [0xb8,0xff,0x44,0x00]
# CHECK-NOFP: vrev64.32 q0, q2  @ encoding: [0xb8,0xff,0x44,0x00]
vrev64.32 q0, q2

# CHECK: vrev32.8 q0, q1  @ encoding: [0xb0,0xff,0xc2,0x00]
# CHECK-NOFP: vrev32.8 q0, q1  @ encoding: [0xb0,0xff,0xc2,0x00]
vrev32.8 q0, q1

# CHECK: vrev32.16 q0, q5  @ encoding: [0xb4,0xff,0xca,0x00]
# CHECK-NOFP: vrev32.16 q0, q5  @ encoding: [0xb4,0xff,0xca,0x00]
vrev32.16 q0, q5

# CHECK: vrev16.8 q0, q2  @ encoding: [0xb0,0xff,0x44,0x01]
# CHECK-NOFP: vrev16.8 q0, q2  @ encoding: [0xb0,0xff,0x44,0x01]
vrev16.8 q0, q2

# CHECK: vmvn q0, q2  @ encoding: [0xb0,0xff,0xc4,0x05]
# CHECK-NOFP: vmvn q0, q2  @ encoding: [0xb0,0xff,0xc4,0x05]
vmvn q0, q2

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.s8 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.s16 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.s32 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.u8 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.u16 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.u32 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.i8 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.i16 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.i32 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.f16 q2, q1, q7

# CHECK: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
# CHECK-NOFP: veor q2, q1, q7  @ encoding: [0x02,0xff,0x5e,0x41]
veor.f32 q2, q1, q7

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.s8 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.s16 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.s32 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.u8 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.u16 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.u32 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.i8 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.i16 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.i32 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.f16 q0, q3, q2

# CHECK: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
# CHECK-NOFP: vorn q0, q3, q2  @ encoding: [0x36,0xef,0x54,0x01]
vorn.f32 q0, q3, q2

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.s8 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.s16 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.s32 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.u8 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.u16 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.u32 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.i8 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.i16 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.i32 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.f16 q1, q2, q1

# CHECK: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
# CHECK-NOFP: vorr q1, q2, q1  @ encoding: [0x24,0xef,0x52,0x21]
vorr.f32 q1, q2, q1

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.s8 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.s16 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.s32 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.u8 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.u16 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.u32 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.i8 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.i16 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.i32 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.f16 q0, q2, q0

# CHECK: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
# CHECK-NOFP: vand q0, q2, q0  @ encoding: [0x04,0xef,0x50,0x01]
vand.f32 q0, q2, q0

# CHECK: vmov.8 q0[1], r8  @ encoding: [0x40,0xee,0x30,0x8b]
# CHECK-NOFP: vmov.8 q0[1], r8  @ encoding: [0x40,0xee,0x30,0x8b]
vmov.8 q0[1], r8

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmov.8 q0[16], r8

# CHECK: vmov.16 q0[2], r5  @ encoding: [0x20,0xee,0x30,0x5b]
# CHECK-NOFP: vmov.16 q0[2], r5  @ encoding: [0x20,0xee,0x30,0x5b]
vmov.16 q0[2], r5

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmov.16 q0[8], r5

# CHECK: vmov.32 q6[3], r11  @ encoding: [0x2d,0xee,0x10,0xbb]
# CHECK-NOFP: vmov.32 q6[3], r11  @ encoding: [0x2d,0xee,0x10,0xbb]
vmov.32 q6[3], r11

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmov.32 q6[4], r11

# CHECK: vmov.32 r0, q1[0]  @ encoding: [0x12,0xee,0x10,0x0b]
# CHECK-NOFP: vmov.32 r0, q1[0]  @ encoding: [0x12,0xee,0x10,0x0b]
vmov.32 r0, q1[0]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmov.32 r0, q1[4]

# CHECK: vmov.s16 r1, q2[7]  @ encoding: [0x35,0xee,0x70,0x1b]
# CHECK-NOFP: vmov.s16 r1, q2[7]  @ encoding: [0x35,0xee,0x70,0x1b]
vmov.s16 r1, q2[7]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmov.s16 r1, q2[8]

# CHECK: vmov.s8 r0, q4[13]  @ encoding: [0x79,0xee,0x30,0x0b]
# CHECK-NOFP: vmov.s8 r0, q4[13]  @ encoding: [0x79,0xee,0x30,0x0b]
vmov.s8 r0, q4[13]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmov.s8 r0, q4[16]

# CHECK: vmov.u16 r0, q1[4]  @ encoding: [0x93,0xee,0x30,0x0b]
# CHECK-NOFP: vmov.u16 r0, q1[4]  @ encoding: [0x93,0xee,0x30,0x0b]
vmov.u16 r0, q1[4]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmov.u16 r0, q1[8]

# CHECK: vmov.u8 r0, q5[7]  @ encoding: [0xfa,0xee,0x70,0x0b]
# CHECK-NOFP: vmov.u8 r0, q5[7]  @ encoding: [0xfa,0xee,0x70,0x0b]
vmov.u8 r0, q5[7]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
vmov.u8 r0, q5[16]

vpste
vmvnt q0, q1
vmvne q0, q1
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vmvnt q0, q1 @ encoding: [0xb0,0xff,0xc2,0x05]
# CHECK-NOFP: vmvnt q0, q1 @ encoding: [0xb0,0xff,0xc2,0x05]
# CHECK: vmvne q0, q1 @ encoding: [0xb0,0xff,0xc2,0x05]
# CHECK-NOFP: vmvne q0, q1 @ encoding: [0xb0,0xff,0xc2,0x05]

vpste
vornt.s8 q0, q1, q2
vorne.s8 q0, q1, q2
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vornt q0, q1, q2 @ encoding: [0x32,0xef,0x54,0x01]
# CHECK-NOFP: vornt q0, q1, q2 @ encoding: [0x32,0xef,0x54,0x01]
# CHECK: vorne q0, q1, q2 @ encoding: [0x32,0xef,0x54,0x01]
# CHECK-NOFP: vorne q0, q1, q2 @ encoding: [0x32,0xef,0x54,0x01]
