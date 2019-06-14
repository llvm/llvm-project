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
