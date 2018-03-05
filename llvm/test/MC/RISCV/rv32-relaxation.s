# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump -d - | FileCheck -check-prefix=INSTR %s

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
#INSTR-NEXT: bnez a0, 326
  c.bnez a0, FAR_BRANCH_NEGATIVE
#INSTR-NEXT: bnez a0, -268
  c.bnez a0, FAR_JUMP
#INSTR-NEXT: bnez a0, 2320
  c.bnez a0, FAR_JUMP_NEGATIVE
#INSTR-NEXT: bnez a0, -2278

  c.beqz a0, NEAR
#INSTR-NEXT: c.beqz a0, 52
  c.beqz a0, NEAR_NEGATIVE
#INSTR-NEXT: c.beqz a0, -24
  c.beqz a0, FAR_BRANCH
#INSTR-NEXT: beqz a0, 306
  c.beqz a0, FAR_BRANCH_NEGATIVE
#INSTR-NEXT: beqz a0, -288
  c.beqz a0, FAR_JUMP
#INSTR-NEXT: beqz a0, 2300
  c.beqz a0, FAR_JUMP_NEGATIVE
#INSTR-NEXT: beqz a0, -2298

  c.j NEAR
#INSTR-NEXT: c.j 32
  c.j NEAR_NEGATIVE
#INSTR-NEXT: c.j -44
  c.j FAR_BRANCH
#INSTR-NEXT: c.j 286
  c.j FAR_BRANCH_NEGATIVE
#INSTR-NEXT: c.j -306
  c.j FAR_JUMP
#INSTR-NEXT: j 2284
  c.j FAR_JUMP_NEGATIVE
#INSTR-NEXT: j -2314

  c.jal NEAR
#INSTR: c.jal 16
  c.jal NEAR_NEGATIVE
#INSTR: c.jal -60
  c.jal FAR_BRANCH
#INSTR-NEXT: c.jal 270
  c.jal FAR_BRANCH_NEGATIVE
#INSTR-NEXT: c.jal -322
  c.jal FAR_JUMP
#INSTR-NEXT: jal 2268
  c.jal FAR_JUMP_NEGATIVE
#INSTR-NEXT: jal -2330

NEAR:
  c.nop
.space 256
FAR_BRANCH:
  c.nop
.space 2000
FAR_JUMP:
  c.nop
