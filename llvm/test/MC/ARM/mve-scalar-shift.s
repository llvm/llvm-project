# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK-NOMVE %s
# RUN:     FileCheck --check-prefix=ERROR-NOMVE < %t %s

# CHECK: asrl    r0, r1, #23  @ encoding: [0x50,0xea,0xef,0x51]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
asrl    r0, r1, #23

# CHECK: asrl    lr, r1, #27  @ encoding: [0x5e,0xea,0xef,0x61]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
asrl    lr, r1, #27

# CHECK: it eq @ encoding: [0x08,0xbf]
# CHECK-NEXT: asrleq    lr, r1, #27  @ encoding: [0x5e,0xea,0xef,0x61]
it eq
asrleq    lr, r1, #27

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid instruction
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: {{error|note}}: invalid instruction
asrl    r3, r2, #33

# ERROR: [[@LINE+3]]:{{[0-9]+}}: {{error|note}}: operand must be an immediate in the range [1,32]
# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r12] or r14
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
asrl    r0, r1, #33

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
asrl    r0, r0, #32

# CHECK: asrl    r0, r1, r4  @ encoding: [0x50,0xea,0x2d,0x41]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
asrl    r0, r1, r4

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: invalid instruction
asrl    r0, r0, r4

# The assembler will reject the above shifts when MVE is not supported,
# so the previous valid instruction will be IT EQ, so we need to add
# a NOPEQ:
nopeq

# CHECK: cinc    lr, r2, lo  @ encoding: [0x52,0xea,0x22,0x9e]
# CHECK-NOMVE: cinc    lr, r2, lo  @ encoding: [0x52,0xea,0x22,0x9e]
csinc   lr, r2, r2, hs

# CHECK: cinc    lr, r7, pl  @ encoding: [0x57,0xea,0x47,0x9e]
# CHECK-NOMVE: cinc    lr, r7, pl  @ encoding: [0x57,0xea,0x47,0x9e]
cinc    lr, r7, pl

# CHECK: cinv    lr, r12, hs  @ encoding: [0x5c,0xea,0x3c,0xae]
# CHECK-NOMVE: cinv    lr, r12, hs  @ encoding: [0x5c,0xea,0x3c,0xae]
cinv    lr, r12, hs

# CHECK: cneg    lr, r10, hs  @ encoding: [0x5a,0xea,0x3a,0xbe]
# CHECK-NOMVE: cneg    lr, r10, hs  @ encoding: [0x5a,0xea,0x3a,0xbe]
csneg   lr, r10, r10, lo

# CHECK: csel    r9, r9, r11, vc  @ encoding: [0x59,0xea,0x7b,0x89]
# CHECK-NOMVE: csel    r9, r9, r11, vc  @ encoding: [0x59,0xea,0x7b,0x89]
csel    r9, r9, r11, vc

# CHECK: cset    lr, eq  @ encoding: [0x5f,0xea,0x1f,0x9e]
# CHECK-NOMVE: cset    lr, eq  @ encoding: [0x5f,0xea,0x1f,0x9e]
cset    lr, eq

# CHECK: csetm   lr, hs  @ encoding: [0x5f,0xea,0x3f,0xae]
# CHECK-NOMVE: csetm   lr, hs  @ encoding: [0x5f,0xea,0x3f,0xae]
csetm   lr, hs

# CHECK: csinc   lr, r10, r7, le  @ encoding: [0x5a,0xea,0xd7,0x9e]
# CHECK-NOMVE: csinc   lr, r10, r7, le  @ encoding: [0x5a,0xea,0xd7,0x9e]
csinc   lr, r10, r7, le

# CHECK: csinv   lr, r5, zr, hs  @ encoding: [0x55,0xea,0x2f,0xae]
# CHECK-NOMVE: csinv   lr, r5, zr, hs  @ encoding: [0x55,0xea,0x2f,0xae]
csinv   lr, r5, zr, hs

# CHECK: cinv    lr, r2, pl  @ encoding: [0x52,0xea,0x42,0xae]
# CHECK-NOMVE: cinv    lr, r2, pl  @ encoding: [0x52,0xea,0x42,0xae]
csinv   lr, r2, r2, mi

# CHECK: csneg   lr, r1, r11, vc  @ encoding: [0x51,0xea,0x7b,0xbe]
# CHECK-NOMVE: csneg   lr, r1, r11, vc  @ encoding: [0x51,0xea,0x7b,0xbe]
csneg   lr, r1, r11, vc

# CHECK: lsll    lr, r1, #11  @ encoding: [0x5e,0xea,0xcf,0x21]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
lsll    lr, r1, #11

# CHECK: lsll    lr, r1, r4  @ encoding: [0x5e,0xea,0x0d,0x41]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
lsll    lr, r1, r4

# CHECK: lsrl    lr, r1, #12  @ encoding: [0x5e,0xea,0x1f,0x31]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
lsrl    lr, r1, #12

# CHECK: sqrshr  lr, r12  @ encoding: [0x5e,0xea,0x2d,0xcf]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
sqrshr  lr, r12

# CHECK: sqrshr  r11, r12  @ encoding: [0x5b,0xea,0x2d,0xcf]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
sqrshr  r11, r12

# CHECK: sqrshrl lr, r3, r8  @ encoding: [0x5f,0xea,0x2d,0x83]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
sqrshrl lr, r3, r8

# CHECK: sqshl   lr, #17  @ encoding: [0x5e,0xea,0x7f,0x4f]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
sqshl   lr, #17

# CHECK: sqshll  lr, r11, #28  @ encoding: [0x5f,0xea,0x3f,0x7b]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
sqshll  lr, r11, #28

# CHECK: srshr   lr, #11  @ encoding: [0x5e,0xea,0xef,0x2f]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
srshr   lr, #11

# CHECK: srshrl  lr, r11, #23  @ encoding: [0x5f,0xea,0xef,0x5b]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
srshrl  lr, r11, #23

# CHECK: uqrshl  lr, r1  @ encoding: [0x5e,0xea,0x0d,0x1f]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
uqrshl  lr, r1

# CHECK: uqrshll lr, r1, r4  @ encoding: [0x5f,0xea,0x0d,0x41]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
uqrshll lr, r1, r4

# CHECK: uqshl   r0, #1  @ encoding: [0x50,0xea,0x4f,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
uqshl   r0, #1

# CHECK: uqshll  lr, r7, #7  @ encoding: [0x5f,0xea,0xcf,0x17]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
uqshll  lr, r7, #7

# CHECK: urshr   r0, #10  @ encoding: [0x50,0xea,0x9f,0x2f]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
urshr   r0, #10

# CHECK: urshrl  r0, r9, #29  @ encoding: [0x51,0xea,0x5f,0x79]
# ERROR-NOMVE: [[@LINE+1]]:{{[0-9]+}}: error: instruction requires: mve
urshrl  r0, r9, #29
