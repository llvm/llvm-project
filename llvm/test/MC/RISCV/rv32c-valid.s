# RUN: llvm-mc %s -triple=riscv32 -mattr=+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump --mattr=+c -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefixes=CHECK-OBJ,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc %s -triple=riscv64 -mattr=+c -riscv-no-aliases -show-encoding \
# RUN:     | FileCheck -check-prefixes=CHECK-ASM,CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump --mattr=+c -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-ASM-AND-OBJ %s

# TODO: more exhaustive testing of immediate encoding.

# CHECK-ASM-AND-OBJ: c.lwsp ra, 0(sp)
# CHECK-ASM: encoding: [0x82,0x40]
c.lwsp ra, 0(sp)
# CHECK-ASM-AND-OBJ: c.swsp ra, 252(sp)
# CHECK-ASM: encoding: [0x86,0xdf]
c.swsp ra, 252(sp)
# CHECK-ASM-AND-OBJ: c.lw a2, 0(a0)
# CHECK-ASM: encoding: [0x10,0x41]
c.lw a2, 0(a0)
# CHECK-ASM-AND-OBJ: c.sw a5, 124(a3)
# CHECK-ASM: encoding: [0xfc,0xde]
c.sw a5, 124(a3)

# CHECK-OBJ: c.j 0xfffff808
# CHECK-ASM: c.j -2048
# CHECK-ASM: encoding: [0x01,0xb0]
c.j -2048
# CHECK-ASM-AND-OBJ: c.jr a7
# CHECK-ASM: encoding: [0x82,0x88]
c.jr a7
# CHECK-ASM-AND-OBJ: c.jalr a1
# CHECK-ASM: encoding: [0x82,0x95]
c.jalr a1
# CHECK-OBJ: c.beqz a3, 0xffffff0e
# CHECK-ASM: c.beqz a3, -256
# CHECK-ASM: encoding: [0x81,0xd2]
c.beqz a3, -256
# CHECK-OBJ: c.bnez a5, 0x10e
# CHECK-ASM: c.bnez a5, 254
# CHECK-ASM: encoding: [0xfd,0xef]
c.bnez a5,  254

# CHECK-ASM-AND-OBJ: c.li a7, 31
# CHECK-ASM: encoding: [0xfd,0x48]
c.li a7, 31
# CHECK-ASM-AND-OBJ: c.addi a3, -32
# CHECK-ASM: encoding: [0x81,0x16]
c.addi a3, -32
# CHECK-ASM-AND-OBJ: c.addi16sp sp, -512
# CHECK-ASM: encoding: [0x01,0x71]
c.addi16sp sp, -512
# CHECK-ASM-AND-OBJ: c.addi16sp sp, 496
# CHECK-ASM: encoding: [0x7d,0x61]
c.addi16sp sp, 496
# CHECK-ASM-AND-OBJ: c.addi4spn a3, sp, 1020
# CHECK-ASM: encoding: [0xf4,0x1f]
c.addi4spn a3, sp, 1020
# CHECK-ASM-AND-OBJ: c.addi4spn a3, sp, 4
# CHECK-ASM: encoding: [0x54,0x00]
c.addi4spn a3, sp, 4
# CHECK-ASM-AND-OBJ: c.slli a1, 1
# CHECK-ASM: encoding: [0x86,0x05]
c.slli a1, 1
# CHECK-ASM-AND-OBJ: c.srli a3, 31
# CHECK-ASM: encoding: [0xfd,0x82]
c.srli a3, 31
# CHECK-ASM-AND-OBJ: c.srai a4, 2
# CHECK-ASM: encoding: [0x09,0x87]
c.srai a4, 2
# CHECK-ASM-AND-OBJ: c.andi a5, 15
# CHECK-ASM: encoding: [0xbd,0x8b]
c.andi a5, 15
# CHECK-ASM-AND-OBJ: c.mv a7, s0
# CHECK-ASM: encoding: [0xa2,0x88]
c.mv a7, s0
# CHECK-ASM-AND-OBJ: c.and a1, a2
# CHECK-ASM: encoding: [0xf1,0x8d]
c.and a1, a2
# CHECK-ASM-AND-OBJ: c.or a2, a3
# CHECK-ASM: encoding: [0x55,0x8e]
c.or a2, a3
# CHECK-ASM-AND-OBJ: c.xor a3, a4
# CHECK-ASM: encoding: [0xb9,0x8e]
c.xor a3, a4
# CHECK-ASM-AND-OBJ: c.sub a4, a5
# CHECK-ASM: encoding: [0x1d,0x8f]
c.sub a4, a5
# CHECK-ASM-AND-OBJ: c.nop
# CHECK-ASM: encoding: [0x01,0x00]
c.nop
# CHECK-ASM-AND-OBJ: c.ebreak
# CHECK-ASM: encoding: [0x02,0x90]
c.ebreak
# CHECK-ASM-AND-OBJ: c.lui s0, 1
# CHECK-ASM: encoding: [0x05,0x64]
c.lui s0, 1
# CHECK-ASM-AND-OBJ: c.lui s0, 31
# CHECK-ASM: encoding: [0x7d,0x64]
c.lui s0, 31
# CHECK-ASM-AND-OBJ: c.lui s0, 1048544
# CHECK-ASM: encoding: [0x01,0x74]
c.lui s0, 0xfffe0
# CHECK-ASM-AND-OBJ: c.lui s0, 1048575
# CHECK-ASM: encoding: [0x7d,0x74]
c.lui s0, 0xfffff
# CHECK-ASM-AND-OBJ: c.unimp
# CHECK-ASM: encoding: [0x00,0x00]
c.unimp
