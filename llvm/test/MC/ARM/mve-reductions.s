# RUN: llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK %s

# CHECK: vabav.s8  r0, q1, q3 @ encoding: [0x82,0xee,0x07,0x0f]
# CHECK-NOFP: vabav.s8  r0, q1, q3 @ encoding: [0x82,0xee,0x07,0x0f]
vabav.s8 r0, q1, q3

# CHECK: vabav.s16  r0, q1, q3 @ encoding: [0x92,0xee,0x07,0x0f]
# CHECK-NOFP: vabav.s16  r0, q1, q3 @ encoding: [0x92,0xee,0x07,0x0f]
vabav.s16 r0, q1, q3

# CHECK: vabav.s32  r0, q1, q3 @ encoding: [0xa2,0xee,0x07,0x0f]
# CHECK-NOFP: vabav.s32  r0, q1, q3 @ encoding: [0xa2,0xee,0x07,0x0f]
vabav.s32 r0, q1, q3

# CHECK: vabav.u8  r0, q1, q3 @ encoding: [0x82,0xfe,0x07,0x0f]
# CHECK-NOFP: vabav.u8  r0, q1, q3 @ encoding: [0x82,0xfe,0x07,0x0f]
vabav.u8 r0, q1, q3

# CHECK: vabav.u16  r0, q1, q3 @ encoding: [0x92,0xfe,0x07,0x0f]
# CHECK-NOFP: vabav.u16  r0, q1, q3 @ encoding: [0x92,0xfe,0x07,0x0f]
vabav.u16 r0, q1, q3

# CHECK: vabav.u32  r0, q1, q3 @ encoding: [0xa2,0xfe,0x07,0x0f]
# CHECK-NOFP: vabav.u32  r0, q1, q3 @ encoding: [0xa2,0xfe,0x07,0x0f]
vabav.u32 r0, q1, q3
