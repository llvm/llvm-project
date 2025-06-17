# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcibi %s \
# RUN:     -filetype=obj -o - \
# RUN:     | llvm-readelf -sr - \
# RUN:     | FileCheck %s


## This checks that the vendor identifier symbols required for vendor
## relocations do not interfere with symbols with identical names that
## are written in assembly.

  .option exact

  qc.e.bgeui s0, 20, QUALCOMM

  .global QUALCOMM
QUALCOMM:
  nop

  qc.e.bgeui s0, 20, QUALCOMM


# CHECK-LABEL: Relocation section '.rela.text'
## Note the different values for the "Sym. Value" Field
# CHECK: R_RISCV_VENDOR    00000000 QUALCOMM + 0
# CHECK: R_RISCV_CUSTOM193 00000006 QUALCOMM + 0
# CHECK: R_RISCV_VENDOR    00000000 QUALCOMM + 0
# CHECK: R_RISCV_CUSTOM193 00000006 QUALCOMM + 0


# CHECK-LABEL: Symbol table '.symtab'
# CHECK-NOT: QUALCOMM
# CHECK: 00000000 0 NOTYPE  LOCAL  DEFAULT ABS QUALCOMM
# CHECK-NOT: QUALCOMM
# CHECK: 00000006 0 NOTYPE  GLOBAL DEFAULT   2 QUALCOMM
# CHECK-NOT: QUALCOMM
