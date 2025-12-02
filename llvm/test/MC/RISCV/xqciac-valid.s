# Xqciac - Qualcomm uC Load-Store Address Calculation Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciac,+zba -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciac,+zba < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciac,+zba -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciac,+zba -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciac,+zba < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciac,+zba --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-NOALIAS: qc.c.muliadd    a0, a1, 0
# CHECK-ALIAS: qc.c.muliadd    a0, a1, 0
# CHECK-ENC: encoding: [0x8a,0x21]
qc.c.muliadd x10, x11, 0

# CHECK-NOALIAS: qc.c.muliadd    a0, a1, 31
# CHECK-ALIAS: qc.c.muliadd    a0, a1, 31
# CHECK-ENC: encoding: [0xea,0x3d]
qc.c.muliadd x10, x11, 31

# CHECK-NOALIAS: qc.c.muliadd    a0, a1, 16
# CHECK-ALIAS: qc.c.muliadd    a0, a1, 16
# CHECK-ENC: encoding: [0xaa,0x21]
qc.c.muliadd x10, x11, 16


# CHECK-INST: qc.muliadd      tp, t0, 1234
# CHECK-ENC: encoding: [0x0b,0xe2,0x22,0x4d]
qc.muliadd x4, x5, 1234

# CHECK-INST: qc.muliadd      a0, a1, -2048
# CHECK-ENC: encoding: [0x0b,0xe5,0x05,0x80]
qc.muliadd x10, x11, -2048

# CHECK-INST: qc.muliadd      a0, a1, 2047
# CHECK-ENC: encoding: [0x0b,0xe5,0xf5,0x7f]
qc.muliadd x10, x11, 2047


# CHECK-INST: qc.shladd       tp, t0, t1, 12
# CHECK-ENC: encoding: [0x0b,0xb2,0x62,0x58]
qc.shladd x4, x5, x6, 12

# CHECK-INST: qc.shladd       a0, a1, a2, 4
# CHECK-ENC: encoding: [0x0b,0xb5,0xc5,0x48]
qc.shladd x10, x11, x12, 4

# CHECK-INST: qc.shladd       a0, a1, a2, 31
# CHECK-ENC: encoding: [0x0b,0xb5,0xc5,0x7e]
qc.shladd x10, x11, x12, 31

# Check that compress pattern for qc.muliadd works

# CHECK-NOALIAS: qc.c.muliadd    a0, a1, 16
# CHECK-ALIAS: qc.c.muliadd    a0, a1, 16
# CHECK-ENC: encoding: [0xaa,0x21]
qc.muliadd x10, x11, 16

# CHECK-NOALIAS: qc.c.muliadd    a0, a1, 2
# CHECK-ALIAS: qc.c.muliadd    a0, a1, 2
# CHECK-ENC: encoding: [0x8a,0x25]
sh1add x10, x11, x10

# CHECK-NOALIAS: qc.c.muliadd    a0, a1, 4
# CHECK-ALIAS: qc.c.muliadd    a0, a1, 4
# CHECK-ENC: encoding: [0x8a,0x29]
sh2add x10, x11, x10

# CHECK-NOALIAS: qc.c.muliadd    a0, a1, 8
# CHECK-ALIAS: qc.c.muliadd    a0, a1, 8
# CHECK-ENC: encoding: [0x8a,0x31]
sh3add x10, x11, x10
