# Xqcia - Qualcomm uC Arithmetic Extesnsion
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcia -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcia < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcia -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcia -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcia < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcia --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.slasat    a0, gp, a7
# CHECK-ENC: encoding: [0x0b,0xb5,0x11,0x15]
qc.slasat x10, x3, x17

# CHECK-INST: qc.sllsat    s7, s9, s11
# CHECK-ENC: encoding: [0x8b,0xbb,0xbc,0x19]
qc.sllsat x23, x25, x27

# CHECK-INST: qc.addsat    a7, a4, t2
# CHECK-ENC: encoding: [0x8b,0x38,0x77,0x1c]
qc.addsat x17, x14, x7

# CHECK-INST: qc.addusat   s0, s2, t3
# CHECK-ENC: encoding: [0x0b,0x34,0xc9,0x1f]
qc.addusat x8, x18, x28

# CHECK-INST: qc.subsat    s6, sp, a2
# CHECK-ENC: encoding: [0x0b,0x3b,0xc1,0x20]
qc.subsat x22, x2, x12

# CHECK-INST: qc.subusat   s1, a4, a7
# CHECK-ENC: encoding: [0x8b,0x34,0x17,0x23]
qc.subusat x9, x14, x17

# CHECK-INST: qc.wrap  gp, t5, s7
# CHECK-ENC: encoding: [0x8b,0x31,0x7f,0x25]
qc.wrap x3, x30, x23

# CHECK-INST: qc.wrapi   t1, a2, 2047
# CHECK-ENC: encoding: [0x0b,0x03,0xf6,0x7f]
qc.wrapi x6, x12, 2047

# CHECK-INST: qc.norm    gp, t2
# CHECK-ENC: encoding: [0x8b,0xb1,0x03,0x0e]
qc.norm x3, x7

# CHECK-INST: qc.normu   a1, a7
# CHECK-ENC: encoding: [0x8b,0xb5,0x08,0x10]
qc.normu x11, x17

# CHECK-INST: qc.normeu  s10, t6
# CHECK-ENC: encoding: [0x0b,0xbd,0x0f,0x12]
qc.normeu x26, x31
