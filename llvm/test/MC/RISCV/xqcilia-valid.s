# Xqcilia - Qualcomm uC Large Immediate Arithmetic extension
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilia -M no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-NOALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilia < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilia -M no-aliases --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv32 -mattr=+experimental-xqcilia -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ENC,CHECK-INST,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilia < %s \
# RUN:     | llvm-objdump --mattr=+experimental-xqcilia --no-print-imm-hex -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INST %s

.option exact

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

# Check that compress patterns work as expected

.option noexact

# CHECK-ALIAS: addi t0, t0, 20
# CHECK-NOALIAS: c.addi t0, 20
# CHECK-ENC: encoding: [0xd1,0x02]
qc.e.addi x5, x5, 20

# CHECK-ALIAS: andi s1, s1, -10
# CHECK-NOALIAS: c.andi s1, -10
# CHECK-ENC: encoding: [0xd9,0x98]
qc.e.andi x9, x9, -10

# CHECK-ALIAS: mv t0, t1
# CHECK-NOALIAS: c.mv t0, t1
# CHECK-ENC: encoding: [0x9a,0x82]
qc.e.addi x5, x6, 0

# CHECK-ALIAS: addi sp, sp, 48
# CHECK-NOALIAS: c.addi16sp sp, 48
# CHECK-ENC: encoding: [0x45,0x61]
qc.e.addi x2, x2, 48

# CHECK-ALIAS: addi s1, s1, -1
# CHECK-NOALIAS: c.addi s1, -1
# CHECK-ENC: encoding: [0xfd,0x14]
qc.e.addai x9, 4294967295

# CHECK-ALIAS: addi sp, sp, 48
# CHECK-NOALIAS: c.addi16sp sp, 48
# CHECK-ENC: encoding: [0x45,0x61]
qc.e.addai x2, 48

# CHECK-ALIAS: andi s1, s1, -1
# CHECK-NOALIAS: c.andi s1, -1
# CHECK-ENC: encoding: [0xfd,0x98]
qc.e.andai x9, 4294967295

# CHECK-ALIAS: addi t0, t2, 400
# CHECK-NOALIAS: addi t0, t2, 400
# CHECK-ENC: encoding: [0x93,0x82,0x03,0x19]
qc.e.addi x5, x7, 400

# CHECK-ALIAS: andi t0, t2, 750
# CHECK-NOALIAS: andi t0, t2, 750
# CHECK-ENC: encoding: [0x93,0xf2,0xe3,0x2e]
qc.e.andi x5, x7, 750

# CHECK-ALIAS: ori t0, t2, 854
# CHECK-NOALIAS: ori t0, t2, 854
# CHECK-ENC: encoding: [0x93,0xe2,0x63,0x35]
qc.e.ori x5, x7, 854

# CHECK-ALIAS: xori t0, t2, -200
# CHECK-NOALIAS: xori t0, t2, -200
# CHECK-ENC: encoding: [0x93,0xc2,0x83,0xf3]
qc.e.xori x5, x7, -200

# CHECK-ALIAS: addi t2, t2, 400
# CHECK-NOALIAS: addi t2, t2, 400
# CHECK-ENC: encoding: [0x93,0x83,0x03,0x19]
qc.e.addai x7, 400

# CHECK-ALIAS: andi t2, t2, 750
# CHECK-NOALIAS: andi t2, t2, 750
# CHECK-ENC: encoding: [0x93,0xf3,0xe3,0x2e]
qc.e.andai x7, 750

# CHECK-ALIAS: ori t2, t2, 854
# CHECK-NOALIAS: ori t2, t2, 854
# CHECK-ENC: encoding: [0x93,0xe3,0x63,0x35]
qc.e.orai x7, 854

# CHECK-ALIAS: xori t2, t2, -200
# CHECK-NOALIAS: xori t2, t2, -200
# CHECK-ENC: encoding: [0x93,0xc3,0x83,0xf3]
qc.e.xorai x7, -200
