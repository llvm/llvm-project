# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcibi %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcibi %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# FIXUP - Check the fixup on the instruction.
# INSTR - Check the instruction is handled properly by the ASMPrinter.

.text

# Since foo is undefined, this will be relaxed to (qc.beqi + jal)
qc.bnei x6, 10, foo
# RELOC: R_RISCV_JAL foo 0x0
# INSTR: qc.bnei t1, 10, foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_branch

# Since foo is undefined, this will be relaxed to (qc.e.bltui + jal)
qc.e.bgeui x8, 12, foo
# RELOC: R_RISCV_JAL foo 0x0
# INSTR: qc.e.bgeui s0, 12, foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_qc_e_branch

# Check that a label in a different section is handled similar to an undefined
# symbol and gets relaxed to (qc.e.bgeui + jal)
qc.e.bltui x4, 9, .bar
# RELOC: R_RISCV_JAL .bar 0x0
# INSTR: qc.e.bltui tp, 9, .bar
# FIXUP: fixup A - offset: 0, value: .bar, kind: fixup_riscv_qc_e_branch

# Check that branches to a defined symbol are handled correctly
qc.e.beqi x7, 8, .L1
# INSTR: qc.e.beqi t2, 8, .L1
# FIXUP: fixup A - offset: 0, value: .L1, kind: fixup_riscv_qc_e_branch

.L1:
  ret

.section .t2

.bar:
  ret
