# RUN: llvm-mc %s -triple=riscv32 -mattr=+m -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+m -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+m < %s \
# RUN:     | llvm-objdump -mattr=+m -d - | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+m < %s \
# RUN:     | llvm-objdump -mattr=+m -d - | FileCheck -check-prefix=CHECK-INST %s

# CHECK-INST: mul a4, ra, s0
# CHECK: encoding: [0x33,0x87,0x80,0x02]
mul a4, ra, s0
# CHECK-INST: mulh ra, zero, zero
# CHECK: encoding: [0xb3,0x10,0x00,0x02]
mulh x1, x0, x0
# CHECK-INST: mulhsu t0, t2, t1
# CHECK: encoding: [0xb3,0xa2,0x63,0x02]
mulhsu t0, t2, t1
# CHECK-INST: mulhu a5, a4, a3
# CHECK: encoding: [0xb3,0x37,0xd7,0x02]
mulhu a5, a4, a3
# CHECK-INST: div s0, s0, s0
# CHECK: encoding: [0x33,0x44,0x84,0x02]
div s0, s0, s0
# CHECK-INST: divu gp, a0, a1
# CHECK: encoding: [0xb3,0x51,0xb5,0x02]
divu gp, a0, a1
# CHECK-INST: rem s2, s2, s8
# CHECK: encoding: [0x33,0x69,0x89,0x03]
rem s2, s2, s8
# CHECK-INST: remu s2, s2, s8
# CHECK: encoding: [0x33,0x79,0x89,0x03]
remu x18, x18, x24
