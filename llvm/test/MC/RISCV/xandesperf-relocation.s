# RUN: llvm-mc -triple riscv32 -mattr=+xandesperf -M no-aliases < %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+xandesperf < %s \
# RUN:     | llvm-readobj -r - | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# INSTR - Check the instruction is handled properly by the ASMPrinter

.long foo
# RELOC: R_RISCV_32 foo

.quad foo
# RELOC: R_RISCV_64 foo

nds.bbc t0, 7, foo
# RELOC: R_RISCV_CUSTOM241
# INSTR: nds.bbc t0, 7, foo

nds.bbs t0, 7, foo
# RELOC: R_RISCV_CUSTOM241
# INSTR: nds.bbs t0, 7, foo

nds.beqc t0, 7, foo
# RELOC: R_RISCV_CUSTOM241
# INSTR: nds.beqc t0, 7, foo

nds.bnec t0, 7, foo
# RELOC: R_RISCV_CUSTOM241
# INSTR: nds.bnec t0, 7, foo

nds.addigp t0, foo
# RELOC: R_RISCV_CUSTOM246
# INSTR: nds.addigp t0, foo

nds.lbgp t0, foo
# RELOC: R_RISCV_CUSTOM246
# INSTR: nds.lbgp t0, foo

nds.lbugp t0, foo
# RELOC: R_RISCV_CUSTOM246
# INSTR: nds.lbugp t0, foo

nds.lhgp t0, foo
# RELOC: R_RISCV_CUSTOM247
# INSTR: nds.lhgp t0, foo

nds.lhugp t0, foo
# RELOC: R_RISCV_CUSTOM247
# INSTR: nds.lhugp t0, foo

nds.lwgp t0, foo
# RELOC: R_RISCV_CUSTOM248
# INSTR: nds.lwgp t0, foo

nds.sbgp t0, foo
# RELOC: R_RISCV_CUSTOM250
# INSTR: nds.sbgp t0, foo

nds.shgp t0, foo
# RELOC: R_RISCV_CUSTOM251
# INSTR: nds.shgp t0, foo

nds.swgp t0, foo
# RELOC: R_RISCV_CUSTOM252
# INSTR: nds.swgp t0, foo
