# RUN: llvm-mc -triple=x86_64-pc-linux-musl %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux-musl %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

# PRINT:      .reloc 2, R_X86_64_NONE, .data
# PRINT-NEXT: .reloc 1, R_X86_64_NONE, foo+4
# PRINT-NEXT: .reloc 0, R_X86_64_NONE, 8

# CHECK:      0x2 R_X86_64_NONE .data 0x0
# CHECK-NEXT: 0x1 R_X86_64_NONE foo 0x4
# CHECK-NEXT: 0x0 R_X86_64_NONE - 0x8

.text
  ret
  nop
  nop
  .reloc 2, R_X86_64_NONE, .data
  .reloc 1, R_X86_64_NONE, foo+4
  .reloc 0, R_X86_64_NONE, 8

.data
.globl foo
foo:
  .word 0
  .word 0
