# RUN: llvm-mc -triple riscv32 -mattr=+xwchc -show-encoding < %s \
# RUN:   | FileCheck -check-prefixes=CHECK,CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+xwchc -show-encoding \
# RUN:   -M no-aliases < %s | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple riscv32 -mattr=+xwchc -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+xwchc --no-print-imm-hex -d - \
# RUN:   | FileCheck -check-prefixes=CHECK-ALIAS %s
# RUN: llvm-mc -triple riscv32 -mattr=+xwchc -filetype=obj < %s \
# RUN:   | llvm-objdump  --triple=riscv32 --mattr=+xwchc --no-print-imm-hex -d -M no-aliases - \
# RUN:   | FileCheck -check-prefixes=CHECK-INST %s


# CHECK-ALIAS: lbu s0, 5(s1)
# CHECK-INST: qk.c.lbu s0, 5(s1)
# CHECK: # encoding: [0xc0,0x30]
lbu s0, 5(s1)

# CHECK-ALIAS: lbu s0, 31(a0)
# CHECK-INST: qk.c.lbu s0, 31(a0)
# CHECK: # encoding: [0x60,0x3d]
lbu s0, 31(a0)

# CHECK-ALIAS: lbu s0, 0(s2)
# CHECK-INST: lbu s0, 0(s2)
# CHECK: # encoding: [0x03,0x44,0x09,0x00]
lbu s0, 0(s2)

# CHECK-ALIAS: lbu s0, 32(s0)
# CHECK-INST: lbu s0, 32(s0)
# CHECK: # encoding: [0x03,0x44,0x04,0x02]
lbu s0, 32(s0)


# CHECK-ALIAS: sb s0, 5(s1)
# CHECK-INST: qk.c.sb s0, 5(s1)
# CHECK: # encoding: [0xc0,0xb0]
sb s0, 5(s1)

# CHECK-ALIAS: sb s0, 31(a0)
# CHECK-INST: qk.c.sb s0, 31(a0)
# CHECK: # encoding: [0x60,0xbd]
sb s0, 31(a0)

# CHECK-ALIAS: sb s0, 0(s2)
# CHECK-INST: sb s0, 0(s2)
# CHECK: # encoding: [0x23,0x00,0x89,0x00]
sb s0, 0(s2)

# CHECK-ALIAS: sb s0, 32(s0)
# CHECK-INST: sb s0, 32(s0)
# CHECK: # encoding: [0x23,0x00,0x84,0x02]
sb s0, 32(s0)


# CHECK-ALIAS: lhu s0, 10(s1)
# CHECK-INST: qk.c.lhu s0, 10(s1)
# CHECK: # encoding: [0xa2,0x24]
lhu s0, 10(s1)

# CHECK-ALIAS: lhu s0, 62(a0)
# CHECK-INST: qk.c.lhu s0, 62(a0)
# CHECK: # encoding: [0x62,0x3d]
lhu s0, 62(a0)

# CHECK-ALIAS: lhu s0, 0(s2)
# CHECK-INST: lhu s0, 0(s2)
# CHECK: # encoding: [0x03,0x54,0x09,0x00]
lhu s0, 0(s2)

# CHECK-ALIAS: lhu s0, 1(s0)
# CHECK-INST: lhu s0, 1(s0)
# CHECK: # encoding: [0x03,0x54,0x14,0x00]
lhu s0, 1(s0)

# CHECK-ALIAS: lhu s0, 64(s0)
# CHECK-INST: lhu s0, 64(s0)
# CHECK: # encoding: [0x03,0x54,0x04,0x04]
lhu s0, 64(s0)


# CHECK-ALIAS: sh s0, 10(s1)
# CHECK-INST: qk.c.sh s0, 10(s1)
# CHECK: # encoding: [0xa2,0xa4]
sh s0, 10(s1)

# CHECK-ALIAS: sh s0, 62(a0)
# CHECK-INST: qk.c.sh s0, 62(a0)
# CHECK: # encoding: [0x62,0xbd]
sh s0, 62(a0)

# CHECK-ALIAS: sh s0, 0(s2)
# CHECK-INST: sh s0, 0(s2)
# CHECK: # encoding: [0x23,0x10,0x89,0x00]
sh s0, 0(s2)

# CHECK-ALIAS: sh s0, 1(s0)
# CHECK-INST: sh s0, 1(s0)
# CHECK: # encoding: [0xa3,0x10,0x84,0x00]
sh s0, 1(s0)

# CHECK-ALIAS: sh s0, 64(s0)
# CHECK-INST: sh s0, 64(s0)
# CHECK: # encoding: [0x23,0x10,0x84,0x04]
sh s0, 64(s0)


# CHECK-ALIAS: lbu a2, 7(sp)
# CHECK-INST: qk.c.lbusp a2, 7(sp)
# CHECK: # encoding: [0x90,0x83]
lbu a2, 7(sp)

# CHECK-ALIAS: lbu a2, 15(sp)
# CHECK-INST: qk.c.lbusp a2, 15(sp)
# CHECK: # encoding: [0x90,0x87]
lbu a2, 15(sp)

# CHECK-ALIAS: lbu s2, 0(sp)
# CHECK-INST: lbu s2, 0(sp)
# CHECK: # encoding: [0x03,0x49,0x01,0x00]
lbu s2, 0(sp)

# CHECK-ALIAS: lbu s0, 16(sp)
# CHECK-INST: lbu s0, 16(sp)
# CHECK: # encoding: [0x03,0x44,0x01,0x01]
lbu s0, 16(sp)


# CHECK-ALIAS: sb a2, 7(sp)
# CHECK-INST: qk.c.sbsp a2, 7(sp)
# CHECK: # encoding: [0xd0,0x83]
sb a2, 7(sp)

# CHECK-ALIAS: sb a2, 15(sp)
# CHECK-INST: qk.c.sbsp a2, 15(sp)
# CHECK: # encoding: [0xd0,0x87]
sb a2, 15(sp)

# CHECK-ALIAS: sb s2, 0(sp)
# CHECK-INST: sb s2, 0(sp)
# CHECK: # encoding: [0x23,0x00,0x21,0x01]
sb s2, 0(sp)

# CHECK-ALIAS: sb s0, 16(sp)
# CHECK-INST: sb s0, 16(sp)
# CHECK: # encoding: [0x23,0x08,0x81,0x00]
sb s0, 16(sp)


# CHECK-ALIAS: lhu a2, 14(sp)
# CHECK-INST: qk.c.lhusp a2, 14(sp)
# CHECK: # encoding: [0x30,0x87]
lhu a2, 14(sp)

# CHECK-ALIAS: lhu a2, 30(sp)
# CHECK-INST: qk.c.lhusp a2, 30(sp)
# CHECK: # encoding: [0xb0,0x87]
lhu a2, 30(sp)

# CHECK-ALIAS: lhu s2, 0(sp)
# CHECK-INST: lhu s2, 0(sp)
# CHECK: # encoding: [0x03,0x59,0x01,0x00]
lhu s2, 0(sp)

# CHECK-ALIAS: lhu s2, 1(sp)
# CHECK-INST: lhu s2, 1(sp)
# CHECK: # encoding: [0x03,0x59,0x11,0x00]
lhu s2, 1(sp)

# CHECK-ALIAS: lhu s0, 32(sp)
# CHECK-INST: lhu s0, 32(sp)
# CHECK: # encoding: [0x03,0x54,0x01,0x02]
lhu s0, 32(sp)


# CHECK-ALIAS: sh a2, 14(sp)
# CHECK-INST: qk.c.shsp a2, 14(sp)
# CHECK: # encoding: [0x70,0x87]
sh a2, 14(sp)

# CHECK-ALIAS: sh a2, 30(sp)
# CHECK-INST: qk.c.shsp a2, 30(sp)
# CHECK: # encoding: [0xf0,0x87]
sh a2, 30(sp)

# CHECK-ALIAS: sh s2, 0(sp)
# CHECK-INST: sh s2, 0(sp)
# CHECK: # encoding: [0x23,0x10,0x21,0x01]
sh s2, 0(sp)

# CHECK-ALIAS: sh s2, 1(sp)
# CHECK-INST: sh s2, 1(sp)
# CHECK: # encoding: [0xa3,0x10,0x21,0x01]
sh s2, 1(sp)

# CHECK-ALIAS: sh s0, 32(sp)
# CHECK-INST: sh s0, 32(sp)
# CHECK: # encoding: [0x23,0x10,0x81,0x02]
sh s0, 32(sp)
