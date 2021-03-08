# RUN: llvm-mc -triple=riscv32 %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -triple=riscv64 %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -filetype=obj -triple=riscv32 %s | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s | llvm-readobj -r - | FileCheck %s

# PRINT: .reloc 8, R_RISCV_NONE, .data
# PRINT: .reloc 4, R_RISCV_NONE, foo+4
# PRINT: .reloc 0, R_RISCV_NONE, 8
# PRINT: .reloc 0, R_RISCV_32, .data+2
# PRINT: .reloc 0, R_RISCV_SET32, foo+3
# PRINT: .reloc 0, R_RISCV_32_PCREL, 5
# PRINT:      .reloc 0, BFD_RELOC_NONE, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_32, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_64, 9

# CHECK:      0x8 R_RISCV_NONE .data 0x0
# CHECK-NEXT: 0x4 R_RISCV_NONE foo 0x4
# CHECK-NEXT: 0x0 R_RISCV_NONE - 0x8
# CHECK-NEXT: 0x0 R_RISCV_32 .data 0x2
# CHECK-NEXT: 0x0 R_RISCV_SET32 foo 0x3
# CHECK-NEXT: 0x0 R_RISCV_32_PCREL - 0x5
# CHECK-NEXT: 0x0 R_RISCV_NONE - 0x9
# CHECK-NEXT: 0x0 R_RISCV_32 - 0x9
# CHECK-NEXT: 0x0 R_RISCV_64 - 0x9
.text
  ret
  nop
  nop
  .reloc 8, R_RISCV_NONE, .data
  .reloc 4, R_RISCV_NONE, foo+4
  .reloc 0, R_RISCV_NONE, 8

  .reloc 0, R_RISCV_32, .data+2
  .reloc 0, R_RISCV_SET32, foo+3
  .reloc 0, R_RISCV_32_PCREL, 5

  .reloc 0, BFD_RELOC_NONE, 9
  .reloc 0, BFD_RELOC_32, 9
  .reloc 0, BFD_RELOC_64, 9

.data
.globl foo
foo:
  .word 0
  .word 0
  .word 0
