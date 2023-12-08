# RUN: llvm-mc -triple=s390x-linux-gnu %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple=s390x-linux-gnu %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# PRINT:      .reloc 2, R_390_NONE, .data
# PRINT-NEXT: .reloc 1, R_390_NONE, foo+4
# PRINT-NEXT: .reloc 0, R_390_NONE, 8
# PRINT-NEXT: .reloc 0, R_390_64, .data+2
# PRINT-NEXT: .reloc 0, R_390_GOTENT, foo+3
# PRINT-NEXT: .reloc 0, R_390_PC32DBL, 6
# PRINT-NEXT: .reloc 4, R_390_12, foo
# PRINT-NEXT: .reloc 2, R_390_20, foo
# PRINT:      .reloc 0, BFD_RELOC_NONE, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_8, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_16, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_32, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_64, 9

# CHECK:      0x2 R_390_NONE .data 0x0
# CHECK-NEXT: 0x1 R_390_NONE foo 0x4
# CHECK-NEXT: 0x0 R_390_NONE - 0x8
# CHECK-NEXT: 0x0 R_390_64 .data 0x2
# CHECK-NEXT: 0x0 R_390_GOTENT foo 0x3
# CHECK-NEXT: 0x0 R_390_PC32DBL - 0x6
# CHECK-NEXT: 0x4 R_390_12 foo 0x0
# CHECK-NEXT: 0x2 R_390_20 foo 0x0
# CHECK-NEXT: 0x0 R_390_NONE - 0x9
# CHECK-NEXT: 0x0 R_390_8 - 0x9
# CHECK-NEXT: 0x0 R_390_16 - 0x9
# CHECK-NEXT: 0x0 R_390_32 - 0x9
# CHECK-NEXT: 0x0 R_390_64 - 0x9

.text
  br %r14
  nop
  nop
  .reloc 2, R_390_NONE, .data
  .reloc 1, R_390_NONE, foo+4
  .reloc 0, R_390_NONE, 8
  .reloc 0, R_390_64, .data+2
  .reloc 0, R_390_GOTENT, foo+3
  .reloc 0, R_390_PC32DBL, 6
  .reloc 4, R_390_12, foo
  .reloc 2, R_390_20, foo

  .reloc 0, BFD_RELOC_NONE, 9
  .reloc 0, BFD_RELOC_8, 9
  .reloc 0, BFD_RELOC_16, 9
  .reloc 0, BFD_RELOC_32, 9
  .reloc 0, BFD_RELOC_64, 9

.data
.globl foo
foo:
  .word 0
  .word 0
