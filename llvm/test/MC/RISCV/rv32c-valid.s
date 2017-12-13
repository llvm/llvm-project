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

# CHECK-INST: c.li  a7, 31
# CHECK: encoding: [0xfd,0x48]
c.li    a7, 31
# CHECK-INST: c.addi  a3, -32
# CHECK: encoding: [0x81,0x16]
c.addi  a3, -32
# CHECK-INST: c.addi16sp  sp, -512
# CHECK: encoding: [0x01,0x71]
c.addi16sp  sp, -512
# CHECK-INST: c.addi16sp  sp, 496
# CHECK: encoding: [0x7d,0x61]
c.addi16sp  sp, 496
# CHECK-INST: c.addi4spn  a3, sp, 1020
# CHECK: encoding: [0xf4,0x1f]
c.addi4spn      a3, sp, 1020
# CHECK-INST: c.addi4spn  a3, sp, 4
# CHECK: encoding: [0x54,0x00]
c.addi4spn      a3, sp, 4
# CHECK-INST: c.slli  a1, 1
# CHECK: encoding: [0x86,0x05]
c.slli  a1, 1
# CHECK-INST: c.srli  a3, 31
# CHECK: encoding: [0xfd,0x82]
c.srli  a3, 31
# CHECK-INST: c.srai  a4, 2
# CHECK: encoding: [0x09,0x87]
c.srai  a4, 2
# CHECK-INST: c.andi  a5, 15
# CHECK: encoding: [0xbd,0x8b]
c.andi  a5, 15
# CHECK-INST: c.mv    a7, s0
# CHECK: encoding: [0xa2,0x88]
c.mv    a7, s0
# CHECK-INST: c.and   a1, a2
# CHECK: encoding: [0xf1,0x8d]
c.and   a1, a2
# CHECK-INST: c.or    a2, a3
# CHECK: encoding: [0x55,0x8e]
c.or    a2, a3
# CHECK-INST: c.xor   a3, a4
# CHECK: encoding: [0xb9,0x8e]
c.xor   a3, a4
# CHECK-INST: c.sub   a4, a5
# CHECK: encoding: [0x1d,0x8f]
c.sub   a4, a5
# CHECK-INST: c.nop
# CHECK: encoding: [0x01,0x00]
c.nop
# CHECK-INST: c.ebreak
# CHECK: encoding: [0x02,0x90]
c.ebreak
# CHECK-INST: c.lui   s0, 1
# CHECK: encoding: [0x05,0x64]
c.lui   s0, 1
# CHECK-INST: c.lui   s0, 63
# CHECK: encoding: [0x7d,0x74]
c.lui   s0, 63
