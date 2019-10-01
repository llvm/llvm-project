# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s 2> %t \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: FileCheck --check-prefix=ERROR-NOFP %s < %t
# RUN: llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK %s

# CHECK: vmaxnm.f32 q0, q1, q4  @ encoding: [0x02,0xff,0x58,0x0f]
# CHECK-NOFP-NOT: vmaxnm.f32 q0, q1, q4  @ encoding: [0x02,0xff,0x58,0x0f]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vmaxnm.f32 q0, q1, q4

# CHECK: vminnm.f16 q3, q0, q1  @ encoding: [0x30,0xff,0x52,0x6f]
# CHECK-NOFP-NOT: vminnm.f16 q3, q0, q1  @ encoding: [0x30,0xff,0x52,0x6f]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vminnm.f16 q3, q0, q1

# CHECK: vmin.s8 q3, q0, q7 @ encoding: [0x00,0xef,0x5e,0x66]
# CHECK-NOFP: vmin.s8 q3, q0, q7 @ encoding: [0x00,0xef,0x5e,0x66]
vmin.s8 q3, q0, q7

# CHECK: vmin.s16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x06]
# CHECK-NOFP: vmin.s16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x06]
vmin.s16 q0, q1, q2

# CHECK: vmin.s32 q0, q1, q2 @ encoding: [0x22,0xef,0x54,0x06]
# CHECK-NOFP: vmin.s32 q0, q1, q2 @ encoding: [0x22,0xef,0x54,0x06]
vmin.s32 q0, q1, q2

# CHECK: vmin.u8 q0, q1, q2 @ encoding: [0x02,0xff,0x54,0x06]
# CHECK-NOFP: vmin.u8 q0, q1, q2 @ encoding: [0x02,0xff,0x54,0x06]
vmin.u8 q0, q1, q2

# CHECK: vmin.u16 q0, q1, q2 @ encoding: [0x12,0xff,0x54,0x06]
# CHECK-NOFP: vmin.u16 q0, q1, q2 @ encoding: [0x12,0xff,0x54,0x06]
vmin.u16 q0, q1, q2

# CHECK: vmin.u32 q0, q1, q2 @ encoding: [0x22,0xff,0x54,0x06]
# CHECK-NOFP: vmin.u32 q0, q1, q2 @ encoding: [0x22,0xff,0x54,0x06]
vmin.u32 q0, q1, q2

# CHECK: vmax.s8 q3, q0, q7 @ encoding: [0x00,0xef,0x4e,0x66]
# CHECK-NOFP: vmax.s8 q3, q0, q7 @ encoding: [0x00,0xef,0x4e,0x66]
vmax.s8 q3, q0, q7

# CHECK: vmax.s16 q0, q1, q2 @ encoding: [0x12,0xef,0x44,0x06]
# CHECK-NOFP: vmax.s16 q0, q1, q2 @ encoding: [0x12,0xef,0x44,0x06]
vmax.s16 q0, q1, q2

# CHECK: vmax.s32 q0, q1, q2 @ encoding: [0x22,0xef,0x44,0x06]
# CHECK-NOFP: vmax.s32 q0, q1, q2 @ encoding: [0x22,0xef,0x44,0x06]
vmax.s32 q0, q1, q2

# CHECK: vmax.u8 q0, q1, q2 @ encoding: [0x02,0xff,0x44,0x06]
# CHECK-NOFP: vmax.u8 q0, q1, q2 @ encoding: [0x02,0xff,0x44,0x06]
vmax.u8 q0, q1, q2

# CHECK: vmax.u16 q0, q1, q2 @ encoding: [0x12,0xff,0x44,0x06]
# CHECK-NOFP: vmax.u16 q0, q1, q2 @ encoding: [0x12,0xff,0x44,0x06]
vmax.u16 q0, q1, q2

# CHECK: vmax.u32 q0, q1, q2 @ encoding: [0x22,0xff,0x44,0x06]
# CHECK-NOFP: vmax.u32 q0, q1, q2 @ encoding: [0x22,0xff,0x44,0x06]
vmax.u32 q0, q1, q2

vpste
vmint.s8 q0, q1, q2
vmine.s16 q0, q1, q2
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK-NOFP: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vmint.s8 q0, q1, q2 @ encoding: [0x02,0xef,0x54,0x06]
# CHECK-NOFP: vmint.s8 q0, q1, q2 @ encoding: [0x02,0xef,0x54,0x06]
# CHECK: vmine.s16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x06]
# CHECK-NOFP: vmine.s16 q0, q1, q2 @ encoding: [0x12,0xef,0x54,0x06]
