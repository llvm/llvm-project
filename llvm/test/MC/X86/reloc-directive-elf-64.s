# RUN: llvm-mc -triple=x86_64-pc-linux-musl %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux-musl %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# PRINT:      .reloc 2, R_X86_64_NONE, .data
# PRINT-NEXT: .reloc 1, R_X86_64_NONE, foo+4
# PRINT-NEXT: .reloc 0, R_X86_64_NONE, 8
# PRINT-NEXT: .reloc 0, R_X86_64_64, .data+2
# PRINT-NEXT: .reloc 0, R_X86_64_GOTPCRELX, foo+3
# PRINT-NEXT: .reloc 0, R_X86_64_REX_GOTPCRELX, 5
# PRINT:      .reloc 0, BFD_RELOC_NONE, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_8, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_16, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_32, 9
# PRINT-NEXT: .reloc 0, BFD_RELOC_64, 9

# CHECK:      0x2 R_X86_64_NONE .data 0x0
# CHECK-NEXT: 0x1 R_X86_64_NONE foo 0x4
# CHECK-NEXT: 0x0 R_X86_64_NONE - 0x8
# CHECK-NEXT: 0x0 R_X86_64_64 .data 0x2
# CHECK-NEXT: 0x0 R_X86_64_GOTPCRELX foo 0x3
# CHECK-NEXT: 0x0 R_X86_64_REX_GOTPCRELX - 0x5
# CHECK-NEXT: 0x0 R_X86_64_NONE - 0x9
# CHECK-NEXT: 0x0 R_X86_64_8 - 0x9
# CHECK-NEXT: 0x0 R_X86_64_16 - 0x9
# CHECK-NEXT: 0x0 R_X86_64_32 - 0x9
# CHECK-NEXT: 0x0 R_X86_64_64 - 0x9

.text
  ret
  nop
  nop
  .reloc 2, R_X86_64_NONE, .data
  .reloc 1, R_X86_64_NONE, foo+4
  .reloc 0, R_X86_64_NONE, 8
  .reloc 0, R_X86_64_64, .data+2
  .reloc 0, R_X86_64_GOTPCRELX, foo+3
  .reloc 0, R_X86_64_REX_GOTPCRELX, 5

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
