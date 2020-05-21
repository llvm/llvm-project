# RUN: llvm-mc -triple riscv32 -mattr=+c -riscv-no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c,+relax < %s \
# RUN:     | llvm-readobj -r | FileCheck -check-prefix=RELOC %s

# COM: Check prefixes:
# COM: RELOC - Check the relocation in the object.
# COM: FIXUP - Check the fixup on the instruction.
# COM: INSTR - Check the instruction is handled properly by the ASMPrinter
c.jal foo
# A compressed jump (c.j) to an unresolved symbol will be relaxed to a (jal).
# RELOC: R_RISCV_JAL
# INSTR: c.jal foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_rvc_jump

c.bnez a0, foo
# A compressed branch (c.bnez) to an unresolved symbol will be relaxed to a (bnez).
# RELOC: R_RISCV_BRANCH
# INSTR: c.bnez a0, foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_rvc_branch
