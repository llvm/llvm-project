# RUN: llvm-mc -triple=sparc %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -triple=sparcv9 %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -filetype=obj -triple=sparc %s | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=sparcv9 %s | llvm-readobj -r - | FileCheck %s

# PRINT: .reloc 8, R_SPARC_NONE, .data
# PRINT: .reloc 4, R_SPARC_NONE, foo+4
# PRINT: .reloc 0, R_SPARC_NONE, 8
# PRINT: .reloc 0, R_SPARC_32, .data+2
# PRINT: .reloc 0, R_SPARC_UA16, foo+3
# PRINT: .reloc 0, R_SPARC_DISP32, foo+5
# PRINT:      .reloc 0, BFD_RELOC_NONE, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_32, foo+2
# PRINT-NEXT: .reloc 0, BFD_RELOC_64, foo+3

# CHECK:      0x8 R_SPARC_NONE .data 0x0
# CHECK-NEXT: 0x4 R_SPARC_NONE foo 0x4
# CHECK-NEXT: 0x0 R_SPARC_NONE - 0x8
# CHECK-NEXT: 0x0 R_SPARC_32 .data 0x2
# CHECK-NEXT: 0x0 R_SPARC_UA16 foo 0x3
# CHECK-NEXT: 0x0 R_SPARC_DISP32 foo 0x5
# CHECK-NEXT: 0x0 R_SPARC_NONE - 0x9
# CHECK-NEXT: 0x0 R_SPARC_32 foo 0x2
# CHECK-NEXT: 0x0 R_SPARC_64 foo 0x3
.text
  ret
  nop
  nop
  .reloc 8, R_SPARC_NONE, .data
  .reloc 4, R_SPARC_NONE, foo+4
  .reloc 0, R_SPARC_NONE, 8

  .reloc 0, R_SPARC_32, .data+2
  .reloc 0, R_SPARC_UA16, foo+3
  .reloc 0, R_SPARC_DISP32, foo+5

  .reloc 0, BFD_RELOC_NONE, 9
  .reloc 0, BFD_RELOC_32, foo+2
  .reloc 0, BFD_RELOC_64, foo+3

.data
.globl foo
foo:
  .word 0
  .word 0
  .word 0
