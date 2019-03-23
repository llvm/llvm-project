# RUN: llvm-mc %s -triple=riscv32 -riscv-no-aliases -mattr=+e -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-ASM-AND-OBJ %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+e < %s \
# RUN:     | llvm-objdump -riscv-no-aliases -d -r - \
# RUN:     | FileCheck -check-prefix=CHECK-ASM-AND-OBJ %s

# This file provides a basic sanity check for RV32E, checking that the expected
# set of registers and instructions are accepted.

# CHECK-ASM-AND-OBJ: lui zero, 1
lui x0, 1
# CHECK-ASM-AND-OBJ: auipc ra, 2
auipc x1, 2

# CHECK-ASM-AND-OBJ: jal sp, 4
jal x2, 4
# CHECK-ASM-AND-OBJ: jalr gp, gp, 4
jalr x3, x3, 4

# CHECK-ASM-AND-OBJ: beq tp, t0, 8
beq x4, x5, 8
# CHECK-ASM-AND-OBJ: bne t1, t2, 12
bne x6, x7, 12
# CHECK-ASM-AND-OBJ: blt s0, s1, 16
blt x8, x9, 16
# CHECK-ASM-AND-OBJ: bge a0, a1, 20
bge x10, x11, 20
# CHECK-ASM-AND-OBJ: bgeu a2, a3, 24
bgeu x12, x13, 24

# CHECK-ASM-AND-OBJ: lb a4, 25(a5)
lb x14, 25(x15)
# CHECK-ASM-AND-OBJ: lh zero, 26(ra)
lh zero, 26(ra)
# CHECK-ASM-AND-OBJ: lw sp, 28(gp)
lw sp, 28(gp)
# CHECK-ASM-AND-OBJ: lbu tp, 29(t0)
lbu tp, 29(t0)
# CHECK-ASM-AND-OBJ: lhu t1, 30(t2)
lhu t1, 30(t2)
# CHECK-ASM-AND-OBJ: sb s0, 31(s1)
sb s0, 31(s1)
# CHECK-ASM-AND-OBJ: sh a0, 32(a1)
sh a0, 32(a1)
# CHECK-ASM-AND-OBJ: sw a2, 36(a3)
sw a2, 36(a3)

# CHECK-ASM-AND-OBJ: addi a4, a5, 37
addi a4, a5, 37
# CHECK-ASM-AND-OBJ: slti a0, a2, -20
slti a0, a2, -20
# CHECK-ASM-AND-OBJ: xori tp, t1, -99
xori tp, t1, -99
# CHECK-ASM-AND-OBJ: ori a0, a1, -2048
ori a0, a1, -2048
# CHECK-ASM-AND-OBJ: andi ra, sp, 2047
andi ra, sp, 2047
# CHECK-ASM-AND-OBJ: slli t1, t1, 31
slli t1, t1, 31
# CHECK-ASM-AND-OBJ: srli a0, a4, 0
srli a0, a4, 0
# CHECK-ASM-AND-OBJ: srai a1, sp, 15
srai a1, sp, 15
# CHECK-ASM-AND-OBJ: slli t0, t1, 13
slli t0, t1, 13

# CHECK-ASM-AND-OBJ: add ra, zero, zero
add ra, zero, zero
# CHECK-ASM-AND-OBJ: sub t0, t2, t1
sub t0, t2, t1
# CHECK-ASM-AND-OBJ: sll a5, a4, a3
sll a5, a4, a3
# CHECK-ASM-AND-OBJ: slt s0, s0, s0
slt s0, s0, s0
# CHECK-ASM-AND-OBJ: sltu gp, a0, a1
sltu gp, a0, a1
# CHECK-ASM-AND-OBJ: xor s1, s0, s1
xor s1, s0, s1
# CHECK-ASM-AND-OBJ: srl a0, s0, t0
srl a0, s0, t0
# CHECK-ASM-AND-OBJ: sra t0, a3, zero
sra t0, a3, zero
# CHECK-ASM-AND-OBJ: or a5, t1, ra
or a5, t1, ra
# CHECK-ASM-AND-OBJ: and a0, s1, a3
and a0, s1, a3

# CHECK-ASM-AND-OBJ: fence iorw, iorw
fence iorw, iorw
# CHECK-ASM-AND-OBJ: fence.tso
fence.tso
# CHECK-ASM-AND-OBJ: fence.i
fence.i

# CHECK-ASM-AND-OBJ: ecall
ecall
# CHECK-ASM-AND-OBJ: ebreak
ebreak
# CHECK-ASM-AND-OBJ: unimp
unimp

# CHECK-ASM-AND-OBJ: csrrw t0, 4095, t1
csrrw t0, 0xfff, t1
# CHECK-ASM-AND-OBJ: csrrs s0, cycle, zero
csrrs s0, 0xc00, x0
# CHECK-ASM-AND-OBJ: csrrs s0, fflags, a5
csrrs s0, 0x001, a5
# CHECK-ASM-AND-OBJ: csrrc sp, ustatus, ra
csrrc sp, 0x000, ra
# CHECK-ASM-AND-OBJ: csrrwi a5, ustatus, 0
csrrwi a5, 0x000, 0
# CHECK-ASM-AND-OBJ: csrrsi t2, 4095, 31
csrrsi t2, 0xfff, 31
# CHECK-ASM-AND-OBJ: csrrci t1, sscratch, 5
csrrci t1, 0x140, 5
