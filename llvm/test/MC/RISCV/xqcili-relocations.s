# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcili %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcili %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# FIXUP - Check the fixup on the instruction.
# INSTR - Check the instruction is handled properly by the ASMPrinter.

.text

qc.li x4, %qc.abs20(foo)
# RELOC: R_RISCV_CUSTOM192 foo 0x0
# INSTR: qc.li tp, %qc.abs20(foo)
# FIXUP: fixup A - offset: 0, value: %qc.abs20(foo), kind: fixup_riscv_qc_abs20_u

qc.e.li x5, foo
# RELOC: R_RISCV_CUSTOM194 foo 0x0
# INSTR: qc.e.li t0, foo
# FIXUP: fixup A - offset: 0, value: foo, kind: fixup_riscv_qc_e_32

# Check that a label in a different section is handled similar to an undefined symbol
qc.li x9, %qc.abs20(.bar)
# RELOC: R_RISCV_CUSTOM192 .bar 0x0
# INSTR: qc.li s1, %qc.abs20(.bar)
# FIXUP: fixup A - offset: 0, value: %qc.abs20(.bar), kind: fixup_riscv_qc_abs20_u

qc.e.li x8, .bar
# RELOC: R_RISCV_CUSTOM194 .bar 0x0
# INSTR: qc.e.li s0, .bar
# FIXUP: fixup A - offset: 0, value: .bar, kind: fixup_riscv_qc_e_32

# Check that branches to a defined symbol are handled correctly
qc.li x7, %qc.abs20(.L1)
# INSTR: qc.li t2, %qc.abs20(.L1)
# FIXUP: fixup A - offset: 0, value: %qc.abs20(.L1), kind: fixup_riscv_qc_abs20_u

qc.e.li x6, .L1
# INSTR: qc.e.li t1, .L1
# FIXUP: fixup A - offset: 0, value: .L1, kind: fixup_riscv_qc_e_32

.L1:
  ret

.section .t2

.bar:
  ret
