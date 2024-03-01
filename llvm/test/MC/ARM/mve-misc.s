# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve -show-encoding  < %s \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -mattr=+mve.fp,+fp64 -show-encoding  < %s 2>%t \
# RUN:   | FileCheck --check-prefix=CHECK %s
# RUN:     FileCheck --check-prefix=ERROR < %t %s
# RUN: not llvm-mc -triple=thumbv8.1m.main-none-eabi -show-encoding  < %s 2>%t
# RUN:     FileCheck --check-prefix=ERROR-NOMVE < %t %s

# CHECK: vpsel   q0, q5, q2  @ encoding: [0x3b,0xfe,0x05,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vpsel   q0, q5, q2

# CHECK: vpnot  @ encoding: [0x31,0xfe,0x4d,0x0f]
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vpnot

# CHECK: wlstp.8     lr, r0, #1668  @ encoding: [0x00,0xf0,0x43,0xc3]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r0, #1668

# CHECK: wlstp.16     lr, r0, #1668  @ encoding: [0x10,0xf0,0x43,0xc3]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.16     lr, r0, #1668

# CHECK: wlstp.32     lr, r4, #2706  @ encoding: [0x24,0xf0,0x49,0xcd]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.32     lr, r4, #2706

# CHECK: wlstp.64     lr, lr, #3026  @ encoding: [0x3e,0xf0,0xe9,0xcd]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.64     lr, lr, #3026

# CHECK: wlstp.8     lr, r5, #3436  @ encoding: [0x05,0xf0,0xb7,0xc6]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r5, #3436

# CHECK: wlstp.16     lr, r1, #1060  @ encoding: [0x11,0xf0,0x13,0xc2]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.16     lr, r1, #1060

# CHECK: wlstp.32     lr, r7, #4036  @ encoding: [0x27,0xf0,0xe3,0xc7]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.32     lr, r7, #4036

# CHECK: wlstp.8     lr, r1, #538  @ encoding: [0x01,0xf0,0x0d,0xc9]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r1, #538

# CHECK: wlstp.8     lr, r10, #1404  @ encoding: [0x0a,0xf0,0xbf,0xc2]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r10, #1404

# CHECK: wlstp.8     lr, r10, #1408  @ encoding: [0x0a,0xf0,0xc1,0xc2]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r10, #1408

# CHECK: wlstp.8     lr, r10, #2358  @ encoding: [0x0a,0xf0,0x9b,0xcc]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r10, #2358

# CHECK: wlstp.8     lr, r10, #4086  @ encoding: [0x0a,0xf0,0xfb,0xcf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r10, #4086

# CHECK: wlstp.8     lr, r11, #1442  @ encoding: [0x0b,0xf0,0xd1,0xca]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r11, #1442

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: loop end is out of range or not a positive multiple of 2
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
wlstp.8     lr, r10, #1443

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: loop end is out of range or not a positive multiple of 2
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
wlstp.8     lr, r10, #4096

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r12] or r14
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
wlstp.8     lr, sp, #1442

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r12] or r14
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
wlstp.16     lr, sp, #1442

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
wlstp.32     r10, r11, #1442

# CHECK: wlstp.8     lr, r1, .Lendloop  @ encoding: [0x01'A',0xf0'A',0x01'A',0xc0'A']
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.8     lr, r1, .Lendloop

# CHECK: wlstp.16     lr, r2, .Lendloop  @ encoding: [0x12'A',0xf0'A',0x01'A',0xc0'A']
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.16     lr, r2, .Lendloop

# CHECK: wlstp.32     lr, r3, .Lendloop  @ encoding: [0x23'A',0xf0'A',0x01'A',0xc0'A']
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.32     lr, r3, .Lendloop

# CHECK: wlstp.64     lr, r5, .Lendloop  @ encoding: [0x35'A',0xf0'A',0x01'A',0xc0'A']
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.64     lr, r5, .Lendloop

# CHECK: wlstp.64     lr, r5, #0  @ encoding: [0x35,0xf0,0x01,0xc0]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
wlstp.64     lr, r5, #0

# CHECK: dlstp.8     lr, r5  @ encoding: [0x05,0xf0,0x01,0xe0]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
dlstp.8     lr, r5

# CHECK: dlstp.16     lr, r5  @ encoding: [0x15,0xf0,0x01,0xe0]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
dlstp.16     lr, r5

# CHECK: dlstp.32     lr, r7  @ encoding: [0x27,0xf0,0x01,0xe0]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
dlstp.32     lr, r7

# CHECK: dlstp.64     lr, r2  @ encoding: [0x32,0xf0,0x01,0xe0]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
dlstp.64     lr, r2

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r12] or r14
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
dlstp.64     lr, sp

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
dlstp.64     r10, r0

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: operand must be a register in range [r0, r12] or r14
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
dlstp.64     lr, pc

# CHECK: letp lr, #-2 @ encoding: [0x1f,0xf0,0x01,0xc8]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
letp lr, #-2

# CHECK: letp lr, #-8 @ encoding: [0x1f,0xf0,0x05,0xc0]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
letp lr, #-8

# CHECK: letp lr, #-4094 @ encoding: [0x1f,0xf0,0xff,0xcf]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
letp lr, #-4094

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: invalid operand for instruction
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
letp r0, #-8

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: loop start is out of range or not a negative multiple of 2
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
letp lr, #8

# ERROR: [[@LINE+2]]:{{[0-9]+}}: {{error|note}}: loop start is out of range or not a negative multiple of 2
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
letp lr, #-4096

# CHECK: letp lr, .Lstartloop @ encoding: [0x1f'A',0xf0'A',0x01'A',0xc0'A']
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
letp lr, .Lstartloop

# CHECK: lctp @ encoding: [0x0f,0xf0,0x01,0xe0]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
lctp

# CHECK: it eq @ encoding: [0x08,0xbf]
it eq
# CHECK: lctpeq @ encoding: [0x0f,0xf0,0x01,0xe0]
# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
lctpeq

# ERROR-NOMVE: [[@LINE+1]]:1: error: instruction requires: mve
vpste
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vpselt.s16 q0, q1, q2
# ERROR-NOMVE: [[@LINE+1]]:1: error: invalid instruction
vpsele.i32 q0, q1, q2
# CHECK: vpste @ encoding: [0x71,0xfe,0x4d,0x8f]
# CHECK: vpselt q0, q1, q2 @ encoding: [0x33,0xfe,0x05,0x0f]
# CHECK: vpsele q0, q1, q2 @ encoding: [0x33,0xfe,0x05,0x0f]
