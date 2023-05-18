# RUN: llvm-mc -triple=avr %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -filetype=obj -triple=avr %s | llvm-readobj -r - | FileCheck %s

# PRINT:      .reloc 4, R_AVR_NONE, .data
# PRINT-NEXT: .reloc 2, R_AVR_NONE, foo+4
# PRINT-NEXT: .reloc 0, R_AVR_NONE, 8
# PRINT:      .reloc 0, R_AVR_32, .data+2
# PRINT-NEXT: .reloc 0, R_AVR_16, foo+3
# PRINT:      .reloc 0, BFD_RELOC_NONE, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_16, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_32, 9

# CHECK:      Section ({{.*}}) .rela.text {
# CHECK-NEXT:   0x4 R_AVR_NONE .data 0x0
# CHECK-NEXT:   0x2 R_AVR_NONE foo 0x4
# CHECK-NEXT:   0x0 R_AVR_NONE - 0x8
# CHECK-NEXT:   0x0 R_AVR_32 .data 0x2
# CHECK-NEXT:   0x0 R_AVR_16 foo 0x3
# CHECK-NEXT:   0x0 R_AVR_NONE - 0x9
# CHECK-NEXT:   0x0 R_AVR_16 - 0x9
# CHECK-NEXT:   0x0 R_AVR_32 - 0x9
# CHECK-NEXT: }

.text
  ret
  nop
  nop
  .reloc 4, R_AVR_NONE, .data
  .reloc 2, R_AVR_NONE, foo+4
  .reloc 0, R_AVR_NONE, 8

  .reloc 0, R_AVR_32, .data+2
  .reloc 0, R_AVR_16, foo+3

  .reloc 0, BFD_RELOC_NONE, 9
  .reloc 0, BFD_RELOC_16, 9
  .reloc 0, BFD_RELOC_32, 9

.data
.globl foo
foo:
  .word 0
  .word 0
