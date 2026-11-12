# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK-NOFP %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s

# CHECK: vmov    lr, r7, q4[2], q4[0]  @ encoding: [0x07,0xec,0x0e,0x8f]
# CHECK-NOFP: vmov    lr, r7, q4[2], q4[0]  @ encoding: [0x07,0xec,0x0e,0x8f]
vmov    lr, r7, q4[2], q4[0]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Q-registers must be the same
vmov    lr, r7, q5[2], q4[0]

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Q-register indexes must be 2 and 0 or 3 and 1
vmov    lr, r7, q4[2], q4[1]

# CHECK: vmov    q3[3], q3[1], r4, r1  @ encoding: [0x11,0xec,0x14,0x6f]
# CHECK-NOFP: vmov    q3[3], q3[1], r4, r1  @ encoding: [0x11,0xec,0x14,0x6f]
vmov    q3[3], q3[1], r4, r1

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Q-registers must be the same
vmov    q4[3], q3[1], r4, r1

# ERROR: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: Q-register indexes must be 2 and 0 or 3 and 1
vmov    q3[2], q3[1], r4, r1
