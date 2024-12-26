# Xqcisls - Qualcomm uC Scaled Load Store Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisls -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisls < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisls -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcisls -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcisls < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcisls --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.lrb    t0, sp, tp, 4
# CHECK-ENC: encoding: [0x8b,0x72,0x41,0x88]
qc.lrb x5, x2, x4, 4

# CHECK-INST: qc.lrh    ra, a2, t1, 2
# CHECK-ENC: encoding: [0x8b,0x70,0x66,0x94]
qc.lrh x1, x12, x6, 2

# CHECK-INST: qc.lrw    a5, t2, a4, 1
# CHECK-ENC: encoding: [0x8b,0xf7,0xe3,0xa2]
qc.lrw x15, x7, x14, 1

# CHECK-INST: qc.lrbu    s1, a1, tp, 7
# CHECK-ENC: encoding: [0x8b,0xf4,0x45,0xbe]
qc.lrbu x9, x11, x4, 7

# CHECK-INST: qc.lrhu    a6, t1, a0, 4
# CHECK-ENC: encoding: [0x0b,0x78,0xa3,0xc8]
qc.lrhu x16, x6, x10, 4

# CHECK-INST: qc.srb    zero, sp, s0, 3
# CHECK-ENC: encoding: [0x2b,0x60,0x81,0xd6]
qc.srb x0, x2, x8, 3

# CHECK-INST: qc.srh    a3, zero, s4, 6
# CHECK-ENC: encoding: [0xab,0x66,0x40,0xed]
qc.srh x13, x0, x20, 6

# CHECK-INST: qc.srw    a7, s2, s3, 0
# CHECK-ENC: encoding: [0xab,0x68,0x39,0xf1]
qc.srw x17, x18, x19, 0
