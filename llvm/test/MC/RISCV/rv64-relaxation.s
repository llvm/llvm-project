# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c < %s \
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
#INSTR: c.bnez a0, 56
  c.bnez a0, NEAR_NEGATIVE
#INSTR: c.bnez a0, -4
  c.bnez a0, FAR_BRANCH
#INSTR-NEXT: bnez a0, 310
  c.bnez a0, FAR_BRANCH_NEGATIVE
#INSTR-NEXT: bnez a0, -268
  c.bnez a0, FAR_JUMP
#INSTR-NEXT: bnez a0, 2304
  c.bnez a0, FAR_JUMP_NEGATIVE
#INSTR-NEXT: bnez a0, -2278

  c.beqz a0, NEAR
#INSTR-NEXT: c.beqz a0, 36
  c.beqz a0, NEAR_NEGATIVE
#INSTR-NEXT: c.beqz a0, -24
  c.beqz a0, FAR_BRANCH
#INSTR-NEXT: beqz a0, 290
  c.beqz a0, FAR_BRANCH_NEGATIVE
#INSTR-NEXT: beqz a0, -288
  c.beqz a0, FAR_JUMP
#INSTR-NEXT: beqz a0, 2284
  c.beqz a0, FAR_JUMP_NEGATIVE
#INSTR-NEXT: beqz a0, -2298

  c.j NEAR
#INSTR-NEXT: c.j 16
  c.j NEAR_NEGATIVE
#INSTR-NEXT: c.j -44
  c.j FAR_BRANCH
#INSTR-NEXT: c.j 270
  c.j FAR_BRANCH_NEGATIVE
#INSTR-NEXT: c.j -306
  c.j FAR_JUMP
#INSTR-NEXT: j 2268
  c.j FAR_JUMP_NEGATIVE
#INSTR-NEXT: j -2314

NEAR:
  c.nop

.space 256
FAR_BRANCH:
  c.nop

.space 2000
FAR_JUMP:
  c.nop
