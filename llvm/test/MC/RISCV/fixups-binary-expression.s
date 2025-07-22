# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-objdump -M no-aliases -d - \
# RUN:     | FileCheck -check-prefix=CHECK-INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c,+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=CHECK-RELOC %s

.LBB0:

.LBB1:

jal zero, .LBB0+16
# CHECK-INSTR: jal zero, 0x10
# CHECK-RELOC:  R_RISCV_JAL

beq a0, a1, .LBB1+32
# CHECK-INSTR: beq a0, a1, 0x20
# CHECK-RELOC:  R_RISCV_BRANCH

c.j     .+32
# CHECK-INSTR: c.j   0x28
# CHECK-RELOC:  R_RISCV_RVC_JUMP

c.beqz a0, .-2
# CHECK-INSTR: c.beqz a0, 0x8
# CHECK-RELOC:  R_RISCV_RVC_BRANCH
