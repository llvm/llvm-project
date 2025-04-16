# Xqcilo - Qualcomm uC Large Offset Load Store extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilo -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilo < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilo -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilo -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilo < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilo --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.e.lb a1, 0(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0x05,0x00,0x00,0x00]
qc.e.lb x11, (x10)


# CHECK-INST: qc.e.lbu        a1, 0(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0x05,0x40,0x00,0x00]
qc.e.lbu x11, (x10)


# CHECK-INST: qc.e.lh a1, 0(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0x05,0x80,0x00,0x00]
qc.e.lh x11, (x10)


# CHECK-INST: qc.e.lhu        a1, 0(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0x05,0xc0,0x00,0x00]
qc.e.lhu x11, (x10)


# CHECK-INST: qc.e.lw a1, 0(a0)
# CHECK-ENC: encoding: [0x9f,0x65,0x05,0x00,0x00,0x00]
qc.e.lw x11, (x10)


# CHECK-INST: qc.e.sb a1, 0(a0)
# CHECK-ENC: encoding: [0x1f,0x60,0xb5,0x40,0x00,0x00]
qc.e.sb x11, (x10)


# CHECK-INST: qc.e.sh a1, 0(a0)
# CHECK-ENC: encoding: [0x1f,0x60,0xb5,0x80,0x00,0x00]
qc.e.sh x11, (x10)


# CHECK-INST: qc.e.sw a1, 0(a0)
# CHECK-ENC: encoding: [0x1f,0x60,0xb5,0xc0,0x00,0x00]
qc.e.sw x11, (x10)
