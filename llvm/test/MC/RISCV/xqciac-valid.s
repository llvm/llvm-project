# Xqciac - Qualcomm uC Load-Store Address Calculation Extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciac -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciac < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciac -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqciac -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqciac < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqciac --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.c.muladdi    a0, a1, 0
# CHECK-ENC: encoding: [0x8a,0x21]
qc.c.muladdi x10, x11, 0

# CHECK-INST: qc.c.muladdi    a0, a1, 31
# CHECK-ENC: encoding: [0xea,0x3d]
qc.c.muladdi x10, x11, 31

# CHECK-INST: qc.c.muladdi    a0, a1, 16
# CHECK-ENC: encoding: [0xaa,0x21]
qc.c.muladdi x10, x11, 16


# CHECK-INST: qc.muladdi      tp, t0, 1234
# CHECK-ENC: encoding: [0x0b,0xe2,0x22,0x4d]
qc.muladdi x4, x5, 1234

# CHECK-INST: qc.muladdi      a0, a1, -2048
# CHECK-ENC: encoding: [0x0b,0xe5,0x05,0x80]
qc.muladdi x10, x11, -2048

# CHECK-INST: qc.muladdi      a0, a1, 2047
# CHECK-ENC: encoding: [0x0b,0xe5,0xf5,0x7f]
qc.muladdi x10, x11, 2047


# CHECK-INST: qc.shladd       tp, t0, t1, 12
# CHECK-ENC: encoding: [0x0b,0xb2,0x62,0x58]
qc.shladd x4, x5, x6, 12

# CHECK-INST: qc.shladd       a0, a1, a2, 4
# CHECK-ENC: encoding: [0x0b,0xb5,0xc5,0x48]
qc.shladd x10, x11, x12, 4

# CHECK-INST: qc.shladd       a0, a1, a2, 31
# CHECK-ENC: encoding: [0x0b,0xb5,0xc5,0x7e]
qc.shladd x10, x11, x12, 31
