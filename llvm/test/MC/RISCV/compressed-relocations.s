# RUN: llvm-mc -triple riscv32 -mattr=+c -M no-aliases %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+c,+relax < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefixes=RELOC,RELAX %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# INSTR - Check the instruction is handled properly by the ASMPrinter
c.jal foo
# A compressed jump (c.j) to an unresolved symbol will be relaxed to a (jal).
# RELOC: R_RISCV_JAL
# RELAX-NEXT: R_RISCV_RELAX
# INSTR: c.jal foo

c.bnez a0, foo
# A compressed branch (c.bnez) to an unresolved symbol will be relaxed to a (bnez).
# The (bnez) to an unresolved symbol will in turn be relaxed to (beqz; jal)
# RELOC-NEXT: R_RISCV_JAL
# RELAX-NEXT: R_RISCV_RELAX
# INSTR: c.bnez a0, foo
