# RUN: llvm-mc -triple xtensa --mattr=+density < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-FIXUP %s
# RUN: llvm-mc -filetype=obj -triple xtensa --mattr=+density < %s \
# RUN:     | llvm-objdump --mattr=+density -d - | FileCheck -check-prefix=CHECK-INSTR %s


# Checks that fixups that can be resolved within the same object file are
# applied correctly
.align 4
LBL0:

.fill 12

beqz.n a2, LBL1
# CHECK-FIXUP: fixup A - offset: 0, value: LBL1, kind: fixup_xtensa_branch_6
# CHECK-INSTR: beqz.n a2, . +29

beq a0, a1, LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_branch_8
# CHECK-INSTR: beq a0, a1, . -14

beq a0, a1, LBL1
# CHECK-FIXUP: fixup A - offset: 0, value: LBL1, kind: fixup_xtensa_branch_8
# CHECK-INSTR: beq a0, a1, . +24

beqz a2, LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_branch_12
# CHECK-INSTR: beqz a2, . -20

beqz a2, LBL1
# CHECK-FIXUP: fixup A - offset: 0, value: LBL1, kind: fixup_xtensa_branch_12
# CHECK-INSTR: beqz a2, . +18

call0 LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_call_18
# CHECK-INSTR: call0 . -24

call0 LBL2
# CHECK-FIXUP: fixup A - offset: 0, value: LBL2, kind: fixup_xtensa_call_18
# CHECK-INSTR: call0 . +2068

j LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_jump_18
# CHECK-INSTR: j . -32

j LBL2
# CHECK-FIXUP: fixup A - offset: 0, value: LBL2, kind: fixup_xtensa_jump_18
# CHECK-INSTR: j . +2061

l32r a1, LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_l32r_16
# CHECK-INSTR: l32r a1, . -38

LBL1:

.fill 2041

.align 4
LBL2:
