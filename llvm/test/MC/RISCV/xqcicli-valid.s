# Xqcicli - Qualcomm uC Conditional Load Immediate Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcicli -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcicli < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcicli -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcicli -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcicli < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcicli --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.lieq   sp, tp, t1, 10
# CHECK-ENC: encoding: [0x5b,0x01,0x62,0x52]
qc.lieq x2, x4, x6, 10

# CHECK-INST: qc.lieqi  t2, ra, 15, 12
# CHECK-ENC: encoding: [0xdb,0x83,0xf0,0x66]
qc.lieqi x7, x1, 15, 12

# CHECK-INST: qc.lige   tp, s0, s4, 2
# CHECK-ENC: encoding: [0x5b,0x52,0x44,0x13]
qc.lige x4, x8, x20, 2

# CHECK-INST: qc.ligei  a7, a1, -4, 9
# CHECK-ENC: encoding: [0xdb,0xd8,0xc5,0x4f]
qc.ligei x17, x11, -4, 9

# CHECK-INST: qc.ligeu  sp, tp, t1, 10
# CHECK-ENC: encoding: [0x5b,0x71,0x62,0x52]
qc.ligeu x2, x4, x6, 10

# CHECK-INST: qc.ligeui sp, a2, 7, -12
# CHECK-ENC: encoding: [0x5b,0x71,0x76,0xa6]
qc.ligeui x2, x12, 7, -12

# CHECK-INST: qc.lilt   s3, s1, a0, 3
# CHECK-ENC: encoding: [0xdb,0xc9,0xa4,0x1a]
qc.lilt x19, x9, x10, 3

# CHECK-INST: qc.lilti  s1, a1, -14, 2
# CHECK-ENC: encoding: [0xdb,0xc4,0x25,0x17]
qc.lilti x9, x11, -14, 2

# CHECK-INST: qc.liltu  ra, s3, a2, 13
# CHECK-ENC: encoding: [0xdb,0xe0,0xc9,0x6a]
qc.liltu x1, x19, x12, 13

# CHECK-INST: qc.liltui gp, s9, 31, 12
# CHECK-ENC: encoding: [0xdb,0xe1,0xfc,0x67]
qc.liltui x3, x25, 31, 12

# CHECK-INST: qc.line   s2, a4, t1, 10
# CHECK-ENC: encoding: [0x5b,0x19,0x67,0x52]
qc.line x18, x14, x6, 10

# CHECK-INST: qc.linei  t0, ra, 10, 12
# CHECK-ENC: encoding: [0xdb,0x92,0xa0,0x66]
qc.linei x5, x1, 10, 12
