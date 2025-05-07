# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcilb %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilb %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELOC %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# INSTR - Check the instruction is handled properly by the ASMPrinter.

.text

qc.e.j foo
# RELOC: R_RISCV_CUSTOM195 foo 0x0
# INSTR: qc.e.j foo

qc.e.jal foo
# RELOC: R_RISCV_CUSTOM195 foo 0x0
# INSTR: qc.e.jal foo

# Check that a label in a different section is handled similar to an undefined symbol
qc.e.j .bar
# RELOC: R_RISCV_CUSTOM195 .bar 0x0
# INSTR: qc.e.j .bar

qc.e.jal .bar
# RELOC: R_RISCV_CUSTOM195 .bar 0x0
# INSTR: qc.e.jal .bar

# Check that jumps to a defined symbol are handled correctly
qc.e.j .L1
# INSTR:qc.e.j .L1

qc.e.jal .L1
# INSTR:qc.e.jal .L1

.L1:
  ret

.section .t2

.bar:
  ret
