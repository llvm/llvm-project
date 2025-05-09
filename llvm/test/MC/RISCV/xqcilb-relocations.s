# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcilb %s -show-encoding \
# RUN:     | FileCheck -check-prefix=INSTR %s
# RUN: llvm-mc -filetype=obj -triple riscv32 -mattr=+experimental-xqcilb %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck -check-prefix=RELOC %s
# RUN: llvm-readelf -s %t.o | FileCheck -check-prefix=VENDORSYM %s

# Check prefixes:
# RELOC - Check the relocation in the object.
# INSTR - Check the instruction is handled properly by the ASMPrinter.
# VENDORSYM - Check the vendor symbol.

.text

qc.e.j foo
# RELOC: R_RISCV_VENDOR QUALCOMM 0x0
# RELOC-NEXT: R_RISCV_CUSTOM195 foo 0x0
# INSTR: qc.e.j foo

qc.e.jal foo
# RELOC: R_RISCV_VENDOR QUALCOMM 0x0
# RELOC-NEXT: R_RISCV_CUSTOM195 foo 0x0
# INSTR: qc.e.jal foo

# Check that a label in a different section is handled similar to an undefined symbol
qc.e.j .bar
# RELOC: R_RISCV_VENDOR QUALCOMM 0x0
# RELOC-NEXT: R_RISCV_CUSTOM195 .bar 0x0
# INSTR: qc.e.j .bar

qc.e.jal .bar
# RELOC: R_RISCV_VENDOR QUALCOMM 0x0
# RELOC-NEXT: R_RISCV_CUSTOM195 .bar 0x0
# INSTR: qc.e.jal .bar

# Check that jumps to a defined symbol are handled correctly
qc.e.j .L1
# INSTR:qc.e.j .L1

qc.e.jal .L1
# INSTR:qc.e.jal .L1

# Check that there is only one vendor symbol created and that it is local and NOTYPE
# VENDORSYM-COUNT-1: 00000000     0 NOTYPE  LOCAL  DEFAULT     2 QUALCOMM
# VENDORSYM-NOT: 00000000     0 NOTYPE  LOCAL  DEFAULT     2 QUALCOMM

.L1:
  ret

.section .t2

.bar:
  ret
