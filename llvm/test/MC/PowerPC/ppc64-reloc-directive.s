# RUN: llvm-mc -triple=powerpc64-linux-musl %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -triple=powerpc64le-linux-musl %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple=powerpc64-linux-musl %s | llvm-readobj -r - | FileCheck %s
# RUN: llvm-mc -filetype=obj -triple=powerpc64le-linux-musl %s | llvm-readobj -r - | FileCheck %s

# PRINT: .reloc 8, R_PPC64_NONE, .data
# PRINT: .reloc 4, R_PPC64_NONE, foo+4
# PRINT: .reloc 0, R_PPC64_NONE, 8
# PRINT: .reloc 0, R_PPC64_ADDR32, .data+2
# PRINT: .reloc 0, R_PPC64_REL16_HI, foo+3
# PRINT: .reloc 0, R_PPC64_REL16_HA, 5
# PRINT: .reloc 0, BFD_RELOC_NONE, 9
# PRINT: .reloc 0, BFD_RELOC_16, 9
# PRINT: .reloc 0, BFD_RELOC_32, 9
# PRINT: .reloc 0, BFD_RELOC_64, 9

# CHECK:      0x8 R_PPC64_NONE .data 0x0
# CHECK-NEXT: 0x4 R_PPC64_NONE foo 0x4
# CHECK-NEXT: 0x0 R_PPC64_NONE - 0x8
# CHECK-NEXT: 0x0 R_PPC64_ADDR32 .data 0x2
# CHECK-NEXT: 0x0 R_PPC64_REL16_HI foo 0x3
# CHECK-NEXT: 0x0 R_PPC64_REL16_HA - 0x5
# CHECK-NEXT: 0x0 R_PPC64_NONE - 0x9
# CHECK-NEXT: 0x0 R_PPC64_ADDR16 - 0x9
# CHECK-NEXT: 0x0 R_PPC64_ADDR32 - 0x9
# CHECK-NEXT: 0x0 R_PPC64_ADDR64 - 0x9

.text
  blr
  nop
  nop
  .reloc 8, R_PPC64_NONE, .data
  .reloc 4, R_PPC64_NONE, foo+4
  .reloc 0, R_PPC64_NONE, 8
  .reloc 0, R_PPC64_ADDR32, .data+2
  .reloc 0, R_PPC64_REL16_HI, foo+3
  .reloc 0, R_PPC64_REL16_HA, 5

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
