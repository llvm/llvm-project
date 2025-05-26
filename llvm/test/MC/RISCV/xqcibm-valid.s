# Xqcibm - Qualcomm uC Bit Manipulation Extension
# Zbs is needed for checking compress instructions patterns for bexti/bseti
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcibm,+zbs -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcibm,+zbs  < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcibm,+zbs -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcibm,+zbs  -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcibm,+zbs < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcibm,+zbs --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.compress2   t2, t0
# CHECK-ENC: encoding: [0x8b,0xb3,0x02,0x00]
qc.compress2 x7, x5

# CHECK-INST: qc.compress3   a0, s6
# CHECK-ENC: encoding: [0x0b,0x35,0x0b,0x02]
qc.compress3 x10, x22

# CHECK-INST: qc.expand2 s7, s7
# CHECK-ENC: encoding: [0x8b,0xbb,0x0b,0x04]
qc.expand2 x23, x23

# CHECK-INST: qc.expand3 sp, t1
# CHECK-ENC: encoding: [0x0b,0x31,0x03,0x06]
qc.expand3 x2, x6

# CHECK-INST: qc.clo s7, s8
# CHECK-ENC: encoding: [0x8b,0x3b,0x0c,0x08]
qc.clo x23, x24

# CHECK-INST: qc.cto a2, a3
# CHECK-ENC: encoding: [0x0b,0xb6,0x06,0x0a]
qc.cto x12, x13

# CHECK-INST: qc.brev32  s4, s8
# CHECK-ENC: encoding: [0x0b,0x3a,0x0c,0x0c]
qc.brev32 x20, x24

# CHECK-INST: qc.insbri  a0, s4, -1024
# CHECK-ENC: encoding: [0x0b,0x05,0x0a,0xc0]
qc.insbri x10, x20, -1024

# CHECK-INST: qc.insbi   t1, -10, 32, 15
# CHECK-ENC: encoding: [0x0b,0x13,0xfb,0x3e]
qc.insbi x6, -10, 32, 15

# CHECK-INST: qc.insb    a0, t2, 6, 31
# CHECK-ENC: encoding: [0x0b,0x95,0xf3,0x4b]
qc.insb x10, x7, 6, 31

# CHECK-INST: qc.insbh   s4, a2, 8, 12
# CHECK-ENC: encoding: [0x0b,0x1a,0xc6,0x8e]
qc.insbh x20, x12, 8, 12

# CHECK-INST: qc.extu    a5, a2, 20, 20
# CHECK-ENC: encoding: [0x8b,0x27,0x46,0x27]
qc.extu x15, x12, 20, 20

# CHECK-INST: qc.ext    s11, t1, 31, 1
# CHECK-ENC: encoding: [0x8b,0x2d,0x13,0x7c]
qc.ext x27, x6, 31, 1

# CHECK-INST: qc.extdu   ra, s0, 32, 8
# CHECK-ENC: encoding: [0x8b,0x20,0x84,0xbe]
qc.extdu x1, x8, 32, 8

# CHECK-INST: qc.extd   a3, s5, 10, 15
# CHECK-ENC: encoding: [0x8b,0xa6,0xfa,0xd2]
qc.extd x13, x21, 10, 15

# CHECK-INST: qc.insbr a0, s3, t0
# CHECK-ENC: encoding: [0x0b,0xb5,0x59,0x00]
qc.insbr x10, x19, x5

# CHECK-INST: qc.insbhr    a5, tp, t1
# CHECK-ENC: encoding: [0x8b,0x37,0x62,0x02]
qc.insbhr x15, x4, x6

# CHECK-INST: qc.insbpr    s5, s0, s1
# CHECK-ENC: encoding: [0x8b,0x3a,0x94,0x04]
qc.insbpr x21, x8, x9

# CHECK-INST: qc.insbprh   sp, gp, a1
# CHECK-ENC: encoding: [0x0b,0xb1,0xb1,0x06]
qc.insbprh x2, x3, x11

# CHECK-INST: qc.extdur    s1, s3, t4
# CHECK-ENC: encoding: [0x8b,0xb4,0xd9,0x09]
qc.extdur x9, x19, x29

# CHECK-INST: qc.extdr    a2, t4, t5
# CHECK-ENC: encoding: [0x0b,0xb6,0xee,0x0b]
qc.extdr x12, x29, x30

# CHECK-INST: qc.extdupr   a3, s7, gp
# CHECK-ENC: encoding: [0x8b,0xb6,0x3b,0x0c]
qc.extdupr x13, x23, x3

# CHECK-INST: qc.extduprh  s2, s0, s1
# CHECK-ENC: encoding: [0x0b,0x39,0x94,0x0e]
qc.extduprh x18, x8, x9

# CHECK-INST: qc.extdpr   ra, tp, a5
# CHECK-ENC: encoding: [0x8b,0x30,0xf2,0x10]
qc.extdpr x1, x4, x15

# CHECK-INST: qc.extdprh  t1, s8, s9
# CHECK-ENC: encoding: [0x0b,0x33,0x9c,0x13]
qc.extdprh x6, x24, x25

# CHECK-NOALIAS: qc.c.bexti  s1, 8
# CHECK-ALIAS: bexti s1, s1, 8
# CHECK-ENC: encoding: [0xa1,0x90]
qc.c.bexti x9, 8

# CHECK-NOALIAS: qc.c.bseti a2, 16
# CHECK-ALIAS: bseti a2, a2, 16
# CHECK-ENC: encoding: [0x41,0x96]
qc.c.bseti x12, 16

# CHECK-NOALIAS: qc.c.extu a5, 32
# CHECK-ALIAS: qc.extu a5, a5, 32, 0
# CHECK-ENC: encoding: [0xfe,0x17]
qc.c.extu x15, 32

# Check that compress patterns work as expected

# CHECK-NOALIAS: qc.c.extu  a1, 11
# CHECK-ALIAS: qc.extu a1, a1, 11, 0
# CHECK-ENC: encoding: [0xaa,0x15]
qc.extu x11, x11, 11, 0

# CHECK-NOALIAS: qc.c.bexti  a1, 5
# CHECK-ALIAS: bexti   a1, a1, 5
# CHECK-ENC: encoding: [0x95,0x91]
qc.extu x11, x11, 1, 5

# CHECK-NOALIAS: qc.c.bexti  s1, 8
# CHECK-ALIAS: bexti s1, s1, 8
# CHECK-ENC-ZBS: encoding: [0xa1,0x90]
bexti x9, x9, 8

# CHECK-NOALIAS: qc.c.bseti a2, 16
# CHECK-ALIAS: bseti a2, a2, 16
# CHECK-ENC: encoding: [0x41,0x96]
bseti x12, x12, 16
