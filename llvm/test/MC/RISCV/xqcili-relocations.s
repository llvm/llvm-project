# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcili %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcili %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# INSTR - Check the instruction is handled properly by the ASMPrinter.

.text

.option exact

qc.li x4, %qc.abs20(foo)
# RELOC: R_RISCV_CUSTOM192 foo 0x0
# INSTR: qc.li tp, %qc.abs20(foo)

qc.e.li x5, foo
# RELOC: R_RISCV_CUSTOM194 foo 0x0
# INSTR: qc.e.li t0, foo

# Check that a label in a different section is handled similar to an undefined symbol
qc.li x9, %qc.abs20(.bar)
# RELOC: R_RISCV_CUSTOM192 .bar 0x0
# INSTR: qc.li s1, %qc.abs20(.bar)

qc.e.li x8, .bar
# RELOC: R_RISCV_CUSTOM194 .bar 0x0
# INSTR: qc.e.li s0, .bar

# Check that branches to a defined symbol are handled correctly
qc.li x7, %qc.abs20(.L1)
# INSTR: qc.li t2, %qc.abs20(.L1)

qc.e.li x6, .L1
# INSTR: qc.e.li t1, .L1

.option noexact

.L1:
  ret

.section .t2

.bar:
  ret
