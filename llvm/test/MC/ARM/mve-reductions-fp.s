# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN:     FileCheck --check-prefix=ERROR-NOFP < %t %s
# RUN: llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding < %s \
# RUN:   | FileCheck --check-prefix=CHECK %s

# CHECK: vminnmv.f16 lr, q3  @ encoding: [0xee,0xfe,0x86,0xef]
# CHECK-NOFP-NOT: vminnmv.f16 lr, q3  @ encoding: [0xee,0xfe,0x86,0xef]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vminnmv.f16 lr, q3

# CHECK: vminnmv.f32 lr, q1  @ encoding: [0xee,0xee,0x82,0xef]
# CHECK-NOFP-NOT: vminnmv.f32 lr, q1  @ encoding: [0xee,0xee,0x82,0xef]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vminnmv.f32 lr, q1

# CHECK: vminnmav.f16 lr, q0  @ encoding: [0xec,0xfe,0x80,0xef]
# CHECK-NOFP-NOT: vminnmav.f16 lr, q0  @ encoding: [0xec,0xfe,0x80,0xef]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vminnmav.f16 lr, q0

# CHECK: vminnmav.f32 lr, q3  @ encoding: [0xec,0xee,0x86,0xef]
# CHECK-NOFP-NOT: vminnmav.f32 lr, q3  @ encoding: [0xec,0xee,0x86,0xef]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vminnmav.f32 lr, q3

# CHECK: vmaxnmv.f16 lr, q1  @ encoding: [0xee,0xfe,0x02,0xef]
# CHECK-NOFP-NOT: vmaxnmv.f16 lr, q1  @ encoding: [0xee,0xfe,0x02,0xef]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vmaxnmv.f16 lr, q1

# CHECK: vmaxnmv.f32 r10, q1  @ encoding: [0xee,0xee,0x02,0xaf]
# CHECK-NOFP-NOT: vmaxnmv.f32 r10, q1  @ encoding: [0xee,0xee,0x02,0xaf]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vmaxnmv.f32 r10, q1

# CHECK: vmaxnmav.f16 r0, q6  @ encoding: [0xec,0xfe,0x0c,0x0f]
# CHECK-NOFP-NOT: vmaxnmav.f16 r0, q6  @ encoding: [0xec,0xfe,0x0c,0x0f]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vmaxnmav.f16 r0, q6

# CHECK: vmaxnmav.f32 lr, q7  @ encoding: [0xec,0xee,0x0e,0xef]
# CHECK-NOFP-NOT: vmaxnmav.f32 lr, q7  @ encoding: [0xec,0xee,0x0e,0xef]
# ERROR-NOFP: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve.fp
vmaxnmav.f32 lr, q7

# ----------------------------------------------------------------------
# The following tests have to go last because of the NOFP-NOT checks inside the
# VPT block.

# CHECK: vpte.i8 eq, q0, q0
# CHECK: vminnmavt.f32 lr, q3  @ encoding: [0xec,0xee,0x86,0xef]
# CHECK-NOFP-NOT: vminnmavt.f32 lr, q3  @ encoding: [0xec,0xee,0x86,0xef]
# CHECK: vminnmave.f32 lr, q3  @ encoding: [0xec,0xee,0x86,0xef]
# CHECK-NOFP-NOT: vminnmave.f32 lr, q3  @ encoding: [0xec,0xee,0x86,0xef]
vpte.i8 eq, q0, q0
vminnmavt.f32 lr, q3
vminnmave.f32 lr, q3
