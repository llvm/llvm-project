# RUN: llvm-mc -triple=aarch64-linux-musl %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-musl %s | llvm-readobj -r - | FileCheck %s

# PRINT: .reloc 8, R_AARCH64_NONE, .data
# PRINT: .reloc 4, R_AARCH64_NONE, foo+4
# PRINT: .reloc 0, R_AARCH64_NONE, 8
# PRINT: .reloc 0, R_AARCH64_ABS64, .data+2
# PRINT: .reloc 0, R_AARCH64_TLSDESC, foo+3
# PRINT: .reloc 0, R_AARCH64_IRELATIVE, 5
# PRINT: .reloc 0, BFD_RELOC_NONE, 9
# PRINT: .reloc 0, BFD_RELOC_16, 9
# PRINT: .reloc 0, BFD_RELOC_32, 9
# PRINT: .reloc 0, BFD_RELOC_64, 9
.text
  ret
  nop
  nop
  .reloc 8, R_AARCH64_NONE, .data
  .reloc 4, R_AARCH64_NONE, foo+4
  .reloc 0, R_AARCH64_NONE, 8

  .reloc 0, R_AARCH64_ABS64, .data+2
  .reloc 0, R_AARCH64_TLSDESC, foo+3
  .reloc 0, R_AARCH64_IRELATIVE, 5

  .reloc 0, BFD_RELOC_NONE, 9
  .reloc 0, BFD_RELOC_16, 9
  .reloc 0, BFD_RELOC_32, 9
  .reloc 0, BFD_RELOC_64, 9

.data
.globl foo
foo:
  .word 0
  .word 0
  .word 0

# CHECK:      0x8 R_AARCH64_NONE .data 0x0
# CHECK-NEXT: 0x4 R_AARCH64_NONE foo 0x4
# CHECK-NEXT: 0x0 R_AARCH64_NONE - 0x8
# CHECK-NEXT: 0x0 R_AARCH64_ABS64 .data 0x2
# CHECK-NEXT: 0x0 R_AARCH64_TLSDESC foo 0x3
# CHECK-NEXT: 0x0 R_AARCH64_IRELATIVE - 0x5
# CHECK-NEXT: 0x0 R_AARCH64_NONE - 0x9
# CHECK-NEXT: 0x0 R_AARCH64_ABS16 - 0x9
# CHECK-NEXT: 0x0 R_AARCH64_ABS32 - 0x9
# CHECK-NEXT: 0x0 R_AARCH64_ABS64 - 0x9
