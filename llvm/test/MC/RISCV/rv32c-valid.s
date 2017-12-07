# RUN: llvm-mc -triple=riscv32 -mattr=+c -show-encoding < %s \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -triple=riscv64 -mattr=+c -show-encoding < %s \
# RUN:     | FileCheck -check-prefixes=CHECK,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump -mattr=+c -d - | FileCheck -check-prefix=CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump -mattr=+c -d - | FileCheck -check-prefix=CHECK-INST %s

# TODO: more exhaustive testing of immediate encoding.

# CHECK-INST: c.lwsp  ra, 0(sp)
# CHECK: encoding: [0x82,0x40]
c.lwsp  ra, 0(sp)
# CHECK-INST: c.swsp  ra, 252(sp)
# CHECK: encoding: [0x86,0xdf]
c.swsp  ra, 252(sp)
# CHECK-INST: c.lw    a2, 0(a0)
# CHECK: encoding: [0x10,0x41]
c.lw    a2, 0(a0)
# CHECK-INST: c.sw    a5, 124(a3)
# CHECK: encoding: [0xfc,0xde]
c.sw    a5, 124(a3)

# CHECK-INST: c.j     -2048
# CHECK: encoding: [0x01,0xb0]
c.j     -2048
# CHECK-INST: c.jal    2046
# CHECK: encoding: [0xfd,0x2f]
c.jal    2046
# CHECK-INST: c.jr    a7
# CHECK: encoding: [0x82,0x88]
c.jr    a7
# CHECK-INST: c.jalr  a1
# CHECK: encoding: [0x82,0x95]
c.jalr  a1
# CHECK-INST: c.beqz  a3, -256
# CHECK: encoding: [0x81,0xd2]
c.beqz  a3, -256
# CHECK-INST: c.bnez  a5,  254
# CHECK: encoding: [0xfd,0xef]
c.bnez  a5,  254
