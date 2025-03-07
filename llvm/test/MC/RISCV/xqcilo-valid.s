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

# CHECK-INST: qc.e.lb a1, 12(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0xc5,0x00,0x00,0x00]
qc.e.lb x11, 12(x10)

# CHECK-INST: qc.e.lb a1, -33554432(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0x05,0x00,0x00,0x80]
qc.e.lb x11, -33554432(x10)

# CHECK-INST: qc.e.lb a1, 33554431(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0xf5,0x3f,0xff,0x7f]
qc.e.lb x11, 33554431(x10)


# CHECK-INST: qc.e.lbu        a1, 12(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0xc5,0x40,0x00,0x00]
qc.e.lbu x11, 12(x10)

# CHECK-INST: qc.e.lbu        a1, -33554432(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0x05,0x40,0x00,0x80]
qc.e.lbu x11, -33554432(x10)

# CHECK-INST: qc.e.lbu        a1, 33554431(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0xf5,0x7f,0xff,0x7f]
qc.e.lbu x11, 33554431(x10)


# CHECK-INST: qc.e.lh a1, 12(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0xc5,0x80,0x00,0x00]
qc.e.lh x11, 12(x10)

# CHECK-INST: qc.e.lh a1, -33554432(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0x05,0x80,0x00,0x80]
qc.e.lh x11, -33554432(x10)

# CHECK-INST: qc.e.lh a1, 33554431(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0xf5,0xbf,0xff,0x7f]
qc.e.lh x11, 33554431(x10)


# CHECK-INST: qc.e.lhu        a1, 12(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0xc5,0xc0,0x00,0x00]
qc.e.lhu x11, 12(x10)

# CHECK-INST: qc.e.lhu        a1, -33554432(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0x05,0xc0,0x00,0x80]
qc.e.lhu x11, -33554432(x10)

# CHECK-INST: qc.e.lhu        a1, 33554431(a0)
# CHECK-ENC: encoding: [0x9f,0x55,0xf5,0xff,0xff,0x7f]
qc.e.lhu x11, 33554431(x10)


# CHECK-INST: qc.e.lw a1, 12(a0)
# CHECK-ENC: encoding: [0x9f,0x65,0xc5,0x00,0x00,0x00]
qc.e.lw x11, 12(x10)

# CHECK-INST: qc.e.lw a1, -33554432(a0)
# CHECK-ENC: encoding: [0x9f,0x65,0x05,0x00,0x00,0x80]
qc.e.lw x11, -33554432(x10)

# CHECK-INST: qc.e.lw a1, 33554431(a0)
# CHECK-ENC: encoding: [0x9f,0x65,0xf5,0x3f,0xff,0x7f]
qc.e.lw x11, 33554431(x10)


# CHECK-INST: qc.e.sb a1, 12(a0)
# CHECK-ENC: encoding: [0x1f,0x66,0xb5,0x40,0x00,0x00]
qc.e.sb x11, 12(x10)

# CHECK-INST: qc.e.sb a1, -33554432(a0)
# CHECK-ENC: encoding: [0x1f,0x60,0xb5,0x40,0x00,0x80]
qc.e.sb x11, -33554432(x10)

# CHECK-INST: qc.e.sb a1, 33554431(a0)
# CHECK-ENC: encoding: [0x9f,0x6f,0xb5,0x7e,0xff,0x7f]
qc.e.sb x11, 33554431(x10)


# CHECK-INST: qc.e.sh a1, 12(a0)
# CHECK-ENC: encoding: [0x1f,0x66,0xb5,0x80,0x00,0x00]
qc.e.sh x11, 12(x10)

# CHECK-INST: qc.e.sh a1, -33554432(a0)
# CHECK-ENC: encoding: [0x1f,0x60,0xb5,0x80,0x00,0x80]
qc.e.sh x11, -33554432(x10)

# CHECK-INST: qc.e.sh a1, 33554431(a0)
# CHECK-ENC: encoding: [0x9f,0x6f,0xb5,0xbe,0xff,0x7f]
qc.e.sh x11, 33554431(x10)


# CHECK-INST: qc.e.sw a1, 12(a0)
# CHECK-ENC: encoding: [0x1f,0x66,0xb5,0xc0,0x00,0x00]
qc.e.sw x11, 12(x10)

# CHECK-INST: qc.e.sw a1, -33554432(a0)
# CHECK-ENC: encoding: [0x1f,0x60,0xb5,0xc0,0x00,0x80]
qc.e.sw x11, -33554432(x10)

# CHECK-INST: qc.e.sw a1, 33554431(a0)
# CHECK-ENC: encoding: [0x9f,0x6f,0xb5,0xfe,0xff,0x7f]
qc.e.sw x11, 33554431(x10)
