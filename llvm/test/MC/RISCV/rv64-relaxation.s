# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c < %s \
# RUN:     | llvm-objdump -d -M no-aliases - | FileCheck -check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c,+relax < %s \
# RUN:     | llvm-objdump -d -M no-aliases - | FileCheck -check-prefix=RELAX-INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+c,+relax < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELAX-RELOC %s

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
#RELAX-INSTR: c.bnez a0, 0
#RELAX-RELOC: R_RISCV_RVC_BRANCH
  c.bnez a0, NEAR_NEGATIVE
#INSTR: c.bnez a0, -4
#RELAX-INSTR: c.bnez a0, 0
#RELAX-RELOC: R_RISCV_RVC_BRANCH
  c.bnez a0, FAR_BRANCH
#INSTR-NEXT: bne a0, zero, 310
#RELAX-INSTR-NEXT: bne a0, zero, 0
#RELAX-RELOC: R_RISCV_BRANCH
  c.bnez a0, FAR_BRANCH_NEGATIVE
#INSTR-NEXT: bne a0, zero, -268
#RELAX-INSTR-NEXT: bne a0, zero, 0
#RELAX-RELOC: R_RISCV_BRANCH
  c.bnez a0, FAR_JUMP
#INSTR-NEXT: bne a0, zero, 2304
#RELAX-INSTR-NEXT: bne a0, zero, 0
#RELAX-RELOC: R_RISCV_BRANCH
  c.bnez a0, FAR_JUMP_NEGATIVE
#INSTR-NEXT: bne a0, zero, -2278
#RELAX-INSTR-NEXT: bne a0, zero, 0
#RELAX-RELOC: R_RISCV_BRANCH

  c.beqz a0, NEAR
#INSTR-NEXT: c.beqz a0, 36
#RELAX-INSTR-NEXT: c.beqz a0, 0
#RELAX-RELOC: R_RISCV_RVC_BRANCH
  c.beqz a0, NEAR_NEGATIVE
#INSTR-NEXT: c.beqz a0, -24
#RELAX-INSTR-NEXT: c.beqz a0, 0
#RELAX-RELOC: R_RISCV_RVC_BRANCH
  c.beqz a0, FAR_BRANCH
#INSTR-NEXT: beq a0, zero, 290
#RELAX-INSTR-NEXT: beq a0, zero, 0
#RELAX-RELOC: R_RISCV_BRANCH
  c.beqz a0, FAR_BRANCH_NEGATIVE
#INSTR-NEXT: beq a0, zero, -288
#RELAX-INSTR-NEXT: beq a0, zero, 0
#RELAX-RELOC: R_RISCV_BRANCH
  c.beqz a0, FAR_JUMP
#INSTR-NEXT: beq a0, zero, 2284
#RELAX-INSTR-NEXT: beq a0, zero, 0
#RELAX-RELOC: R_RISCV_BRANCH
  c.beqz a0, FAR_JUMP_NEGATIVE
#INSTR-NEXT: beq a0, zero, -2298
#RELAX-INSTR-NEXT: beq a0, zero, 0
#RELAX-RELOC: R_RISCV_BRANCH

  c.j NEAR
#INSTR-NEXT: c.j 16
#RELAX-INSTR-NEXT: c.j 0
#RELAX-RELOC: R_RISCV_RVC_JUMP
  c.j NEAR_NEGATIVE
#INSTR-NEXT: c.j -44
#RELAX-INSTR-NEXT: c.j 0
#RELAX-RELOC: R_RISCV_RVC_JUMP
  c.j FAR_BRANCH
#INSTR-NEXT: c.j 270
#RELAX-INSTR-NEXT: c.j 0
#RELAX-RELOC: R_RISCV_RVC_JUMP
  c.j FAR_BRANCH_NEGATIVE
#INSTR-NEXT: c.j -306
#RELAX-INSTR-NEXT: c.j 0
#RELAX-RELOC: R_RISCV_RVC_JUMP
  c.j FAR_JUMP
#INSTR-NEXT: jal zero, 2268
#RELAX-INSTR-NEXT: jal zero, 0
#RELAX-RELOC: R_RISCV_JAL
  c.j FAR_JUMP_NEGATIVE
#INSTR-NEXT: jal zero, -2314
#RELAX-INSTR-NEXT: jal zero, 0
#RELAX-RELOC: R_RISCV_JAL

NEAR:
  c.nop

.space 256
FAR_BRANCH:
  c.nop

.space 2000
FAR_JUMP:
  c.nop
