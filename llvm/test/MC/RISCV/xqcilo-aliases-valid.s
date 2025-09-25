# Xqcilo - Qualcomm uC Large Offset Load Store extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilo -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilo < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilo -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilo -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilo < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilo --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: lb a1, 0(a0)
# CHECK-ENC: encoding: [0x83,0x05,0x05,0x00]
qc.e.lb x11, (x10)


# CHECK-INST: lbu a1, 0(a0)
# CHECK-ENC: encoding: [0x83,0x45,0x05,0x00]
qc.e.lbu x11, (x10)


# CHECK-INST: lh a1, 0(a0)
# CHECK-ENC: encoding: [0x83,0x15,0x05,0x00]
qc.e.lh x11, (x10)


# CHECK-INST: lhu a1, 0(a0)
# CHECK-ENC: encoding: [0x83,0x55,0x05,0x00]
qc.e.lhu x11, (x10)


# CHECK-NOALIAS: c.lw a1, 0(a0)
# CHECK-ALIAS: lw a1, 0(a0)
# CHECK-ENC: encoding: [0x0c,0x41]
qc.e.lw x11, (x10)


# CHECK-INST: sb a1, 0(a0)
# CHECK-ENC: encoding: [0x23,0x00,0xb5,0x00]
qc.e.sb x11, (x10)


# CHECK-INST: sh a1, 0(a0)
# CHECK-ENC: encoding: [0x23,0x10,0xb5,0x00]
qc.e.sh x11, (x10)


# CHECK-NOALIAS: c.sw a1, 0(a0)
# CHECK-ALIAS: sw a1, 0(a0)
# CHECK-ENC: encoding: [0x0c,0xc1]
qc.e.sw x11, (x10)
