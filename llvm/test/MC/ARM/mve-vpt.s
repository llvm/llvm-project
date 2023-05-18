# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: VPT predicated instructions must be in VPT block
vabavt.s32 lr, q1, q3

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: VPT predicated instructions must be in VPT block
vabave.u8 r12, q1, q3

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: instructions in IT block must be predicable
it eq
vabav.s16 lr, q1, q3

# CHECK: vpteee.i8 eq, q0, q1 @ encoding: [0x41,0xfe,0x02,0x2f]
# CHECK-NOFP: vpteee.i8 eq, q0, q1 @ encoding: [0x41,0xfe,0x02,0x2f]
vpteee.i8 eq, q0, q1
vabavt.s32 lr, q1, q3
vabave.s32 lr, q1, q3
vabave.s32 lr, q1, q3
vabave.s32 lr, q1, q3

# CHECK: vptttt.s32 gt, q0, q1 @ encoding: [0x21,0xfe,0x03,0x3f]
# CHECK-NOFP: vptttt.s32 gt, q0, q1 @ encoding: [0x21,0xfe,0x03,0x3f]
vptttt.s32 gt, q0, q1
vabavt.u32 lr, q1, q3
vabavt.s32 lr, q1, q3
vabavt.s16 lr, q1, q3
vabavt.s8 lr, q1, q3

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: instruction in VPT block must be predicable
vpt.s8 le, q0, q1
cinc lr, r2, lo

# ----------------------------------------------------------------------
# The following tests have to go last because of the NOFP-NOT checks inside the
# VPT block.

# CHECK: vptete.f16 ne, q0, q1 @ encoding: [0x71,0xfe,0x82,0xef]
# CHECK-NOFP-NOT: vptete.f16 ne, q0, q1 @ encoding: [0x71,0xfe,0x82,0xef]
vptete.f16 ne, q0, q1
vabavt.s32 lr, q1, q3
vabave.u32 lr, q1, q3
vabavt.s32 lr, q1, q3
vabave.s16 lr, q1, q3
# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: VPT predicated instructions must be in VPT block
vabavt.s32 lr, q1, q3

# CHECK: vpte.i8 eq, q0, q0
# CHECK: vmaxnmt.f16 q1, q6, q2  @ encoding: [0x1c,0xff,0x54,0x2f]
# CHECK-NOFP-NOT: vmaxnmt.f16 q1, q6, q2  @ encoding: [0x1c,0xff,0x54,0x2f]
# CHECK-NOFP-NOT: vmaxnme.f16 q1, q6, q2  @ encoding: [0x1c,0xff,0x54,0x2f]
vpte.i8 eq, q0, q0
vmaxnmt.f16 q1, q6, q2
vmaxnme.f16 q1, q6, q2
