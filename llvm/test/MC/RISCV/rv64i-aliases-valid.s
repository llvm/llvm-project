# RUN: llvm-mc %s -triple=riscv64 -riscv-no-aliases \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-INST %s
# RUN: llvm-mc %s -triple=riscv64 \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-ALIAS %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -riscv-no-aliases -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-OBJ-NOALIAS,CHECK-EXPAND,CHECK-INST %s
# RUN: llvm-mc -filetype=obj -triple riscv64 < %s \
# RUN:     | llvm-objdump -d - \
# RUN:     | FileCheck -check-prefixes=CHECK-EXPAND,CHECK-ALIAS %s

# The following check prefixes are used in this test:
# CHECK-INST.....Match the canonical instr (tests alias to instr. mapping)
# CHECK-ALIAS....Match the alias (tests instr. to alias mapping)
# CHECK-EXPAND...Match canonical instr. unconditionally (tests alias expansion)

# TODO ld
# TODO sd

# CHECK-INST: addi a0, zero, 0
# CHECK-ALIAS: mv a0, zero
li x10, 0
# CHECK-EXPAND: addi a0, zero, 1
li x10, 1
# CHECK-EXPAND: addi a0, zero, -1
li x10, -1
# CHECK-EXPAND: addi a0, zero, 2047
li x10, 2047
# CHECK-EXPAND: addi a0, zero, -2047
li x10, -2047
# CHECK-EXPAND: lui a1, 1
# CHECK-EXPAND: addiw a1, a1, -2048
li x11, 2048
# CHECK-EXPAND: addi a1, zero, -2048
li x11, -2048
# CHECK-EXPAND: lui a1, 1
# CHECK-EXPAND: addiw a1, a1, -2047
li x11, 2049
# CHECK-EXPAND: lui a1, 1048575
# CHECK-EXPAND: addiw a1, a1, 2047
li x11, -2049
# CHECK-EXPAND: lui a1, 1
# CHECK-EXPAND: addiw a1, a1, -1
li x11, 4095
# CHECK-EXPAND: lui a1, 1048575
# CHECK-EXPAND: addiw a1, a1, 1
li x11, -4095
# CHECK-EXPAND: lui a2, 1
li x12, 4096
# CHECK-EXPAND: lui a2, 1048575
li x12, -4096
# CHECK-EXPAND: lui a2, 1
# CHECK-EXPAND: addiw a2, a2, 1
li x12, 4097
# CHECK-EXPAND: lui a2, 1048575
# CHECK-EXPAND: addiw a2, a2, -1
li x12, -4097
# CHECK-EXPAND: lui a2, 524288
# CHECK-EXPAND: addiw a2, a2, -1
li x12, 2147483647
# CHECK-EXPAND: lui a2, 524288
# CHECK-EXPAND: addiw a2, a2, 1
li x12, -2147483647
# CHECK-EXPAND: lui a2, 524288
li x12, -2147483648
# CHECK-EXPAND: lui a2, 524288
li x12, -0x80000000

# CHECK-EXPAND: addi a2, zero, 1
# CHECK-EXPAND: slli a2, a2, 31
li x12, 0x80000000
# CHECK-EXPAND: addi a2, zero, 1
# CHECK-EXPAND: slli a2, a2, 32
# CHECK-EXPAND: addi a2, a2, -1
li x12, 0xFFFFFFFF

# CHECK-EXPAND: addi t0, zero, 1
# CHECK-EXPAND: slli t0, t0, 32
li t0, 0x100000000
# CHECK-EXPAND: addi t1, zero, -1
# CHECK-EXPAND: slli t1, t1, 63
li t1, 0x8000000000000000
# CHECK-EXPAND: addi t1, zero, -1
# CHECK-EXPAND: slli t1, t1, 63
li t1, -0x8000000000000000
# CHECK-EXPAND: lui t2, 9321
# CHECK-EXPAND: addiw t2, t2, -1329
# CHECK-EXPAND: slli t2, t2, 35
li t2, 0x1234567800000000
# CHECK-EXPAND: addi t3, zero, 7
# CHECK-EXPAND: slli t3, t3, 36
# CHECK-EXPAND: addi t3, t3, 11
# CHECK-EXPAND: slli t3, t3, 24
# CHECK-EXPAND: addi t3, t3, 15
li t3, 0x700000000B00000F
# CHECK-EXPAND: lui t4, 583
# CHECK-EXPAND: addiw t4, t4, -1875
# CHECK-EXPAND: slli t4, t4, 14
# CHECK-EXPAND: addi t4, t4, -947
# CHECK-EXPAND: slli t4, t4, 12
# CHECK-EXPAND: addi t4, t4, 1511
# CHECK-EXPAND: slli t4, t4, 13
# CHECK-EXPAND: addi t4, t4, -272
li t4, 0x123456789abcdef0
# CHECK-EXPAND: addi t5, zero, -1
li t5, 0xFFFFFFFFFFFFFFFF

# CHECK-EXPAND: addi a0, zero, 1110
li a0, %lo(0x123456)
# CHECK-OBJ-NOALIAS: addi a0, zero, 0
# CHECK-OBJ: R_RISCV_PCREL_LO12
li a0, %pcrel_lo(0x123456)

# CHECK-OBJ-NOALIAS: addi a0, zero, 0
# CHECK-OBJ: R_RISCV_LO12
li a0, %lo(foo)
# CHECK-OBJ-NOALIAS: addi a0, zero, 0
# CHECK-OBJ: R_RISCV_PCREL_LO12
li a0, %pcrel_lo(foo)

.equ CONST, 0x123456
# CHECK-EXPAND: lui a0, 291
# CHECK-EXPAND: addiw a0, a0, 1110
li a0, CONST

.equ CONST, 0x654321
# CHECK-EXPAND: lui a0, 1620
# CHECK-EXPAND: addiw a0, a0, 801
li a0, CONST

# CHECK-INST: subw t6, zero, ra
# CHECK-ALIAS: negw t6, ra
negw x31, x1
# CHECK-INST: addiw t6, ra, 0
# CHECK-ALIAS: sext.w t6, ra
sext.w x31, x1

# The following aliases are accepted as input but the canonical form
# of the instruction will always be printed.
# CHECK-INST: addiw a2, a3, 4
# CHECK-ALIAS: addiw a2, a3, 4
addw a2,a3,4

# CHECK-INST: slliw a2, a3, 4
# CHECK-ALIAS: slliw a2, a3, 4
sllw a2,a3,4

# CHECK-INST: srliw a2, a3, 4
# CHECK-ALIAS: srliw a2, a3, 4
srlw a2,a3,4

# CHECK-INST: sraiw a2, a3, 4
# CHECK-ALIAS: sraiw a2, a3, 4
sraw a2,a3,4

# CHECK-EXPAND: lwu a0, 0(a1)
lwu x10, (x11)
# CHECK-EXPAND: ld a0, 0(a1)
ld x10, (x11)
# CHECK-EXPAND: sd a0, 0(a1)
sd x10, (x11)
