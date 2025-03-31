# RUN: llvm-mc -triple riscv64 -mattr=+xandesperf -M no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv64 -mattr=+xandesperf < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# INSTR - Check the instruction is handled properly by the ASMPrinter

.long foo
# RELOC: R_RISCV_32 foo

.quad foo
# RELOC: R_RISCV_64 foo

nds.lwugp t0, foo
# RELOC: R_RISCV_CUSTOM248
# INSTR: nds.lwugp t0, foo

nds.ldgp t0, foo
# RELOC: R_RISCV_CUSTOM249
# INSTR: nds.ldgp t0, foo

nds.sdgp t0, foo
# RELOC: R_RISCV_CUSTOM253
# INSTR: nds.sdgp t0, foo
