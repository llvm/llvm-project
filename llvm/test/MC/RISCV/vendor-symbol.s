# RUN: llvm-mc -triple riscv32 -mattr=+experimental-xqcibi,+xandesperf %s \
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

  nds.bbc t0, 7, ANDES

  .global ANDES
ANDES:
  nop

  nds.bbs t0, 7, ANDES


# CHECK-LABEL: Relocation section '.rela.text'
## Note the different values for the "Sym. Value" Field
# CHECK: R_RISCV_VENDOR    00000000 QUALCOMM + 0
# CHECK: R_RISCV_CUSTOM193 00000006 QUALCOMM + 0
# CHECK: R_RISCV_VENDOR    00000000 QUALCOMM + 0
# CHECK: R_RISCV_CUSTOM193 00000006 QUALCOMM + 0
# CHECK: R_RISCV_VENDOR    00000000 ANDES + 0
# CHECK: R_RISCV_CUSTOM241 00000014 ANDES + 0
# CHECK: R_RISCV_VENDOR    00000000 ANDES + 0
# CHECK: R_RISCV_CUSTOM241 00000014 ANDES + 0


# CHECK-LABEL: Symbol table '.symtab'
# CHECK-NOT: QUALCOMM
# CHECK-NOT: ANDES
# CHECK: 00000000 0 NOTYPE  LOCAL  DEFAULT ABS QUALCOMM
# CHECK: 00000000 0 NOTYPE  LOCAL  DEFAULT ABS ANDES
# CHECK-NOT: QUALCOMM
# CHECK-NOT: ANDES
# CHECK: 00000006 0 NOTYPE  GLOBAL DEFAULT   2 QUALCOMM
# CHECK: 00000014 0 NOTYPE  GLOBAL DEFAULT   2 ANDES
# CHECK-NOT: QUALCOMM
# CHECK-NOT: ANDES
