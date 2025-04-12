# Xqcilia - Qualcomm uC Large Immediate Arithmetic extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilia -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilia < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilia -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilia -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilia < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilia --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: qc.e.addai      s1, -1
# CHECK-ENC: encoding: [0x9f,0x24,0xff,0xff,0xff,0xff]
qc.e.addai x9, 4294967295

# CHECK-INST: qc.e.addai      s1, -2147483648
# CHECK-ENC: encoding: [0x9f,0x24,0x00,0x00,0x00,0x80]
qc.e.addai x9, -2147483648


# CHECK-INST: qc.e.addi       a0, s1, -33554432
# CHECK-ENC: encoding: [0x1f,0xb5,0x04,0x80,0x00,0x80]
qc.e.addi x10, x9, -33554432

# CHECK-INST: qc.e.addi       a0, s1, 33554431
# CHECK-ENC: encoding: [0x1f,0xb5,0xf4,0xbf,0xff,0x7f]
qc.e.addi x10, x9, 33554431


# CHECK-INST: qc.e.andai      s1, -1
# CHECK-ENC: encoding: [0x9f,0xa4,0xff,0xff,0xff,0xff]
qc.e.andai x9, 4294967295

# CHECK-INST: qc.e.andai      s1, -2147483648
# CHECK-ENC: encoding: [0x9f,0xa4,0x00,0x00,0x00,0x80]
qc.e.andai x9, -2147483648


# CHECK-INST: qc.e.andi       a0, s1, -33554432
# CHECK-ENC: encoding: [0x1f,0xb5,0x04,0xc0,0x00,0x80]
qc.e.andi x10, x9, -33554432

# CHECK-INST: qc.e.andi       a0, s1, 33554431
# CHECK-ENC: encoding: [0x1f,0xb5,0xf4,0xff,0xff,0x7f]
qc.e.andi x10, x9, 33554431


# CHECK-INST: qc.e.orai       s1, -1
# CHECK-ENC: encoding: [0x9f,0x94,0xff,0xff,0xff,0xff]
qc.e.orai x9, 4294967295

# CHECK-INST: qc.e.orai       s1, -2147483648
# CHECK-ENC: encoding: [0x9f,0x94,0x00,0x00,0x00,0x80]
qc.e.orai x9, -2147483648


# CHECK-INST: qc.e.ori        a0, s1, -33554432
# CHECK-ENC: encoding: [0x1f,0xb5,0x04,0x40,0x00,0x80]
qc.e.ori x10, x9, -33554432

# CHECK-INST: qc.e.ori        a0, s1, 33554431
# CHECK-ENC: encoding: [0x1f,0xb5,0xf4,0x7f,0xff,0x7f]
qc.e.ori x10, x9, 33554431


# CHECK-INST: qc.e.xorai      s1, -1
# CHECK-ENC: encoding: [0x9f,0x14,0xff,0xff,0xff,0xff]
qc.e.xorai x9, 4294967295

# CHECK-INST: qc.e.xorai      s1, -2147483648
# CHECK-ENC: encoding: [0x9f,0x14,0x00,0x00,0x00,0x80]
qc.e.xorai x9, -2147483648


# CHECK-INST: qc.e.xori       a0, s1, -33554432
# CHECK-ENC: encoding: [0x1f,0xb5,0x04,0x00,0x00,0x80]
qc.e.xori x10, x9, -33554432

# CHECK-INST: qc.e.xori       a0, s1, 33554431
# CHECK-ENC: encoding: [0x1f,0xb5,0xf4,0x3f,0xff,0x7f]
qc.e.xori x10, x9, 33554431
