# RUN: llvm-mc -triple xtensa < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s
# RUN: llvm-mc -filetype=obj -triple xtensa < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# FIXUP - Check the fixup on the instruction.
# INSTR - Check the instruction is handled properly by the ASMPrinter

.long func
# RELOC: R_XTENSA_32 func

ball a1, a3, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  ball    a1, a3, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bany a8, a13, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bany    a8, a13, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bbc a8, a7, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bbc     a8, a7, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bbci a3, 16, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bbci    a3, 16, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bbs a12, a5, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bbs     a12, a5, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bbsi a3, 16, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bbsi    a3, 16, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bnall a7, a3, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bnall   a7, a3, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bnone a2, a4, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bnone   a2, a4, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

beq a1, a2, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  beq     a1, a2, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

beq a11, a5, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  beq     a11, a5, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

beqi a1, 256, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  beqi    a1, 256, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

beqi a11, -1, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  beqi    a11, -1, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

beqz a8, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  beqz    a8, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_12

bge a14, a2, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bge     a14, a2, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bgei a11, -1, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bgei    a11, -1, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bgei a11, 128, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bgei    a11, 128, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bgeu a14, a2, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bgeu    a14, a2, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bgeui a9, 32768, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bgeui   a9, 32768, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bgeui a7, 65536, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bgeui   a7, 65536, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bgeui a7, 64, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bgeui   a7, 64, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bgez a8, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bgez    a8, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_12

blt a14, a2, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  blt     a14, a2, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

blti a12, -1, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  blti    a12, -1, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

blti a0, 32, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  blti    a0, 32, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bgeu a13, a1, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bgeu    a13, a1, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bltui a7, 16, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bltui   a7, 16, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bltz a6, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bltz    a6, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_12

bne a3, a4, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bne     a3, a4, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bnei a5, 12, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bnei    a5, 12, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_8

bnez a5, func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  bnez    a5, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_branch_12

call0  func
# RELOC: R_XTENSA_SLOT0_OP
# INST:  call0   func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_call_18

j func
# RELOC: R_XTENSA_SLOT0_OP
# INSTR: j func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_jump_18

l32r a6, func
# RELOC: R_XTENSA_SLOT0_OP
# INSTR: l32r    a6, func
# FIXUP: fixup A - offset: 0, value: func, kind: fixup_xtensa_l32r_16
