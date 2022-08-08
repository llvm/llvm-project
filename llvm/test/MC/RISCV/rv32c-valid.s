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

# RUN: not llvm-mc -triple riscv32 \
# RUN:     -riscv-no-aliases -show-encoding < %s 2>&1 \
# RUN:     | FileCheck -check-prefixes=CHECK-NO-EXT %s

# TODO: more exhaustive testing of immediate encoding.

# CHECK-ASM-AND-OBJ: c.lwsp ra, 0(sp)
# CHECK-ASM: encoding: [0x82,0x40]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.lwsp ra, 0(sp)
# CHECK-ASM-AND-OBJ: c.swsp ra, 252(sp)
# CHECK-ASM: encoding: [0x86,0xdf]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.swsp ra, 252(sp)
# CHECK-ASM-AND-OBJ: c.lw a2, 0(a0)
# CHECK-ASM: encoding: [0x10,0x41]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.lw a2, 0(a0)
# CHECK-ASM-AND-OBJ: c.sw a5, 124(a3)
# CHECK-ASM: encoding: [0xfc,0xde]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.sw a5, 124(a3)

# CHECK-ASM-AND-OBJ: c.lwsp s0, 0(sp)
# CHECK-ASM: encoding: [0x02,0x44]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.lwsp x8, (x2)
# CHECK-ASM-AND-OBJ: c.swsp s0, 0(sp)
# CHECK-ASM: encoding: [0x22,0xc0]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.swsp x8, (x2)
# CHECK-ASM-AND-OBJ: c.lw s0, 0(s1)
# CHECK-ASM: encoding: [0x80,0x40]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.lw x8, (x9)
# CHECK-ASM-AND-OBJ: c.sw s0, 0(s1)
# CHECK-ASM: encoding: [0x80,0xc0]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.sw x8, (x9)

# CHECK-OBJ: c.j 0xfffff810
# CHECK-ASM: c.j -2048
# CHECK-ASM: encoding: [0x01,0xb0]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.j -2048
# CHECK-ASM-AND-OBJ: c.jr a7
# CHECK-ASM: encoding: [0x82,0x88]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.jr a7
# CHECK-ASM-AND-OBJ: c.jalr a1
# CHECK-ASM: encoding: [0x82,0x95]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.jalr a1
# CHECK-OBJ: c.beqz a3, 0xffffff16
# CHECK-ASM: c.beqz a3, -256
# CHECK-ASM: encoding: [0x81,0xd2]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.beqz a3, -256
# CHECK-OBJ: c.bnez a5, 0x116
# CHECK-ASM: c.bnez a5, 254
# CHECK-ASM: encoding: [0xfd,0xef]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.bnez a5,  254

# CHECK-ASM-AND-OBJ: c.li a7, 31
# CHECK-ASM: encoding: [0xfd,0x48]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.li a7, 31
# CHECK-ASM-AND-OBJ: c.addi a3, -32
# CHECK-ASM: encoding: [0x81,0x16]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.addi a3, -32
# CHECK-ASM-AND-OBJ: c.addi16sp sp, -512
# CHECK-ASM: encoding: [0x01,0x71]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.addi16sp sp, -512
# CHECK-ASM-AND-OBJ: c.addi16sp sp, 496
# CHECK-ASM: encoding: [0x7d,0x61]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.addi16sp sp, 496
# CHECK-ASM-AND-OBJ: c.addi4spn a3, sp, 1020
# CHECK-ASM: encoding: [0xf4,0x1f]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.addi4spn a3, sp, 1020
# CHECK-ASM-AND-OBJ: c.addi4spn a3, sp, 4
# CHECK-ASM: encoding: [0x54,0x00]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.addi4spn a3, sp, 4
# CHECK-ASM-AND-OBJ: c.slli a1, 1
# CHECK-ASM: encoding: [0x86,0x05]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.slli a1, 1
# CHECK-ASM-AND-OBJ: c.srli a3, 31
# CHECK-ASM: encoding: [0xfd,0x82]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.srli a3, 31
# CHECK-ASM-AND-OBJ: c.srai a4, 2
# CHECK-ASM: encoding: [0x09,0x87]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.srai a4, 2
# CHECK-ASM-AND-OBJ: c.andi a5, 15
# CHECK-ASM: encoding: [0xbd,0x8b]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.andi a5, 15
# CHECK-ASM-AND-OBJ: c.mv a7, s0
# CHECK-ASM: encoding: [0xa2,0x88]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.mv a7, s0
# CHECK-ASM-AND-OBJ: c.and a1, a2
# CHECK-ASM: encoding: [0xf1,0x8d]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.and a1, a2
# CHECK-ASM-AND-OBJ: c.or a2, a3
# CHECK-ASM: encoding: [0x55,0x8e]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.or a2, a3
# CHECK-ASM-AND-OBJ: c.xor a3, a4
# CHECK-ASM: encoding: [0xb9,0x8e]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.xor a3, a4
# CHECK-ASM-AND-OBJ: c.sub a4, a5
# CHECK-ASM: encoding: [0x1d,0x8f]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.sub a4, a5
# CHECK-ASM-AND-OBJ: c.nop
# CHECK-ASM: encoding: [0x01,0x00]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.nop
# CHECK-ASM-AND-OBJ: c.ebreak
# CHECK-ASM: encoding: [0x02,0x90]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.ebreak
# CHECK-ASM-AND-OBJ: c.lui s0, 1
# CHECK-ASM: encoding: [0x05,0x64]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.lui s0, 1
# CHECK-ASM-AND-OBJ: c.lui s0, 31
# CHECK-ASM: encoding: [0x7d,0x64]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.lui s0, 31
# CHECK-ASM-AND-OBJ: c.lui s0, 1048544
# CHECK-ASM: encoding: [0x01,0x74]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.lui s0, 0xfffe0
# CHECK-ASM-AND-OBJ: c.lui s0, 1048575
# CHECK-ASM: encoding: [0x7d,0x74]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.lui s0, 0xfffff
# CHECK-ASM-AND-OBJ: c.unimp
# CHECK-ASM: encoding: [0x00,0x00]
# CHECK-NO-EXT:  error: instruction requires the following: 'C' (Compressed Instructions){{$}}
c.unimp
