# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump -d -riscv-no-aliases - | FileCheck -check-prefix=INSTR %s

FAR_JUMP_NEGATIVE:
  c.nop
.space 2000

FAR_BRANCH_NEGATIVE:
  c.nop
.space 256

NEAR_NEGATIVE:
  c.nop

start:
  c.bnez a0, NEAR
#INSTR: c.bnez a0, 72
  c.bnez a0, NEAR_NEGATIVE
#INSTR: c.bnez a0, -4
  c.bnez a0, FAR_BRANCH
#INSTR-NEXT: bne a0, zero, 326
  c.bnez a0, FAR_BRANCH_NEGATIVE
#INSTR-NEXT: bne a0, zero, -268
  c.bnez a0, FAR_JUMP
#INSTR-NEXT: bne a0, zero, 2320
  c.bnez a0, FAR_JUMP_NEGATIVE
#INSTR-NEXT: bne a0, zero, -2278

  c.beqz a0, NEAR
#INSTR-NEXT: c.beqz a0, 52
  c.beqz a0, NEAR_NEGATIVE
#INSTR-NEXT: c.beqz a0, -24
  c.beqz a0, FAR_BRANCH
#INSTR-NEXT: beq a0, zero, 306
  c.beqz a0, FAR_BRANCH_NEGATIVE
#INSTR-NEXT: beq a0, zero, -288
  c.beqz a0, FAR_JUMP
#INSTR-NEXT: beq a0, zero, 2300
  c.beqz a0, FAR_JUMP_NEGATIVE
#INSTR-NEXT: beq a0, zero, -2298

  c.j NEAR
#INSTR-NEXT: c.j 32
  c.j NEAR_NEGATIVE
#INSTR-NEXT: c.j -44
  c.j FAR_BRANCH
#INSTR-NEXT: c.j 286
  c.j FAR_BRANCH_NEGATIVE
#INSTR-NEXT: c.j -306
  c.j FAR_JUMP
#INSTR-NEXT: jal zero, 2284
  c.j FAR_JUMP_NEGATIVE
#INSTR-NEXT: jal zero, -2314

  c.jal NEAR
#INSTR: c.jal 16
  c.jal NEAR_NEGATIVE
#INSTR: c.jal -60
  c.jal FAR_BRANCH
#INSTR-NEXT: c.jal 270
  c.jal FAR_BRANCH_NEGATIVE
#INSTR-NEXT: c.jal -322
  c.jal FAR_JUMP
#INSTR-NEXT: jal ra, 2268
  c.jal FAR_JUMP_NEGATIVE
#INSTR-NEXT: jal ra, -2330

NEAR:
  c.nop
.space 256
FAR_BRANCH:
  c.nop
.space 2000
FAR_JUMP:
  c.nop
