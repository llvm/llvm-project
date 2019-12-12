# RUN: llvm-mc -triple xtensa < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=CHECK-FIXUP %s
# RUN: llvm-mc -filetype=obj -triple xtensa < %s \
# RUN:     | llvm-objdump -d - | FileCheck -check-prefix=CHECK-INSTR %s


# Checks that fixups that can be resolved within the same object file are
# applied correctly
.align 4
LBL0:

.fill 12

beq a0, a1, LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_branch_8
# CHECK-INSTR: beq a0, a1, . -12

beq a0, a1, LBL1
# CHECK-FIXUP: fixup A - offset: 0, value: LBL1, kind: fixup_xtensa_branch_8
# CHECK-INSTR: beq a0, a1, . +24

beqz a2, LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_branch_12
# CHECK-INSTR: beqz a2, . -18

beqz a2, LBL1
# CHECK-FIXUP: fixup A - offset: 0, value: LBL1, kind: fixup_xtensa_branch_12
# CHECK-INSTR: beqz a2, . +18

call0 LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_call_18
# CHECK-INSTR: call0 . -24

call0 LBL2
# CHECK-FIXUP: fixup A - offset: 0, value: LBL2, kind: fixup_xtensa_call_18
# CHECK-INSTR: call0 . +2056

j LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_jump_18
# CHECK-INSTR: j . -30

j LBL2
# CHECK-FIXUP: fixup A - offset: 0, value: LBL2, kind: fixup_xtensa_jump_18
# CHECK-INSTR: j . +2047

l32r a1, LBL0
# CHECK-FIXUP: fixup A - offset: 0, value: LBL0, kind: fixup_xtensa_l32r_16
# CHECK-INSTR: l32r a1, . -36

LBL1:

.fill 2041

LBL2:
