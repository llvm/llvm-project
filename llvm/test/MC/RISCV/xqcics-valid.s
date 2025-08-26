# Xqcics - Qualcomm uC Conditional Select Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcics -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcics < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcics -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcics -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcics < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcics --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.selecteqi    s1, 5, tp, gp
# CHECK-ENC: encoding: [0xdb,0xa4,0x42,0x1c]
qc.selecteqi x9, 5, x4, x3

# CHECK-INST: qc.selecteqi    s1, -16, tp, gp
# CHECK-ENC: encoding: [0xdb,0x24,0x48,0x1c]
qc.selecteqi x9, -16, x4, x3

# CHECK-INST: qc.selecteqi    s1, 15, tp, gp
# CHECK-ENC: encoding: [0xdb,0xa4,0x47,0x1c]
qc.selecteqi x9, 15, x4, x3


# CHECK-INST: qc.selectieq    s0, tp, gp, 12
# CHECK-ENC: encoding: [0x5b,0x24,0x32,0x62]
qc.selectieq x8, x4, x3, 12

# CHECK-INST: qc.selectieq    s0, tp, gp, -16
# CHECK-ENC: encoding: [0x5b,0x24,0x32,0x82]
qc.selectieq x8, x4, x3, -16

# CHECK-INST: qc.selectieq    s0, tp, gp, 15
# CHECK-ENC: encoding: [0x5b,0x24,0x32,0x7a]
qc.selectieq x8, x4, x3, 15


# CHECK-INST: qc.selectieqi   s1, 11, gp, 12
# CHECK-ENC: encoding: [0xdb,0xa4,0x35,0x66]
qc.selectieqi x9, 11, x3, 12

# CHECK-INST: qc.selectieqi   s1, -16, gp, 12
# CHECK-ENC: encoding: [0xdb,0x24,0x38,0x66]
qc.selectieqi x9, -16, x3, 12

# CHECK-INST: qc.selectieqi   s1, 15, gp, 12
# CHECK-ENC: encoding: [0xdb,0xa4,0x37,0x66]
qc.selectieqi x9, 15, x3, 12

# CHECK-INST: qc.selectieqi   s1, 11, gp, -16
# CHECK-ENC: encoding: [0xdb,0xa4,0x35,0x86]
qc.selectieqi x9, 11, x3, -16

# CHECK-INST: qc.selectieqi   s1, 11, gp, 15
# CHECK-ENC: encoding: [0xdb,0xa4,0x35,0x7e]
qc.selectieqi x9, 11, x3, 15


# CHECK-INST: qc.selectiieq   s1, gp, 11, 12
# CHECK-ENC: encoding: [0xdb,0xa4,0xb1,0x60]
qc.selectiieq x9, x3, 11, 12

# CHECK-INST: qc.selectiieq   s1, gp, -16, 12
# CHECK-ENC: encoding: [0xdb,0xa4,0x01,0x61]
qc.selectiieq x9, x3, -16, 12

# CHECK-INST: qc.selectiieq   s1, gp, 15, 12
# CHECK-ENC: encoding: [0xdb,0xa4,0xf1,0x60]
qc.selectiieq x9, x3, 15, 12

# CHECK-INST: qc.selectiieq   s1, gp, 11, -16
# CHECK-ENC: encoding: [0xdb,0xa4,0xb1,0x80]
qc.selectiieq x9, x3, 11, -16

# CHECK-INST: qc.selectiieq   s1, gp, 11, 15
# CHECK-ENC: encoding: [0xdb,0xa4,0xb1,0x78]
qc.selectiieq x9, x3, 11, 15


# CHECK-INST: qc.selectiine   s0, gp, 10, 11
# CHECK-ENC: encoding: [0x5b,0xb4,0xa1,0x58]
qc.selectiine x8, x3, 10, 11

# CHECK-INST: qc.selectiine   s0, gp, -16, 11
# CHECK-ENC: encoding: [0x5b,0xb4,0x01,0x59]
qc.selectiine x8, x3, -16, 11

# CHECK-INST: qc.selectiine   s0, gp, 15, 11
# CHECK-ENC: encoding: [0x5b,0xb4,0xf1,0x58]
qc.selectiine x8, x3, 15, 11

# CHECK-INST: qc.selectiine   s0, gp, 10, -16
# CHECK-ENC: encoding: [0x5b,0xb4,0xa1,0x80]
qc.selectiine x8, x3, 10, -16

# CHECK-INST: qc.selectiine   s0, gp, 10, 15
# CHECK-ENC: encoding: [0x5b,0xb4,0xa1,0x78]
qc.selectiine x8, x3, 10, 15


# CHECK-INST: qc.selectine    s0, gp, tp, 11
# CHECK-ENC: encoding: [0x5b,0xb4,0x41,0x5a]
qc.selectine x8, x3, x4, 11

# CHECK-INST: qc.selectine    s0, gp, tp, -16
# CHECK-ENC: encoding: [0x5b,0xb4,0x41,0x82]
qc.selectine x8, x3, x4, -16

# CHECK-INST: qc.selectine    s0, gp, tp, 15
# CHECK-ENC: encoding: [0x5b,0xb4,0x41,0x7a]
qc.selectine x8, x3, x4, 15


# CHECK-INST: qc.selectinei   s0, 11, gp, 12
# CHECK-ENC: encoding: [0x5b,0xb4,0x35,0x66]
qc.selectinei x8, 11, x3, 12

# CHECK-INST: qc.selectinei   s0, -16, gp, 12
# CHECK-ENC: encoding: [0x5b,0x34,0x38,0x66]
qc.selectinei x8, -16, x3, 12

# CHECK-INST: qc.selectinei   s0, 15, gp, 12
# CHECK-ENC: encoding: [0x5b,0xb4,0x37,0x66]
qc.selectinei x8, 15, x3, 12

# CHECK-INST: qc.selectinei   s0, 11, gp, -16
# CHECK-ENC: encoding: [0x5b,0xb4,0x35,0x86]
qc.selectinei x8, 11, x3, -16

# CHECK-INST: qc.selectinei   s0, 11, gp, 15
# CHECK-ENC: encoding: [0x5b,0xb4,0x35,0x7e]
qc.selectinei x8, 11, x3, 15


# CHECK-INST: qc.selectnei    s0, 11, gp, t0
# CHECK-ENC: encoding: [0x5b,0xb4,0x35,0x2c]
qc.selectnei x8, 11, x3, x5

# CHECK-INST: qc.selectnei    s0, -16, gp, t0
# CHECK-ENC: encoding: [0x5b,0x34,0x38,0x2c]
qc.selectnei x8, -16, x3, x5

# CHECK-INST: qc.selectnei    s0, 15, gp, t0
# CHECK-ENC: encoding: [0x5b,0xb4,0x37,0x2c]
qc.selectnei x8, 15, x3, x5

