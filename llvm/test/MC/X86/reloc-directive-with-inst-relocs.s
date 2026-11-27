# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux-musl %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s

## Test that .reloc relocations are emitted in groups.

# CHECK:      0x6 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT: 0x3 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT: 0x0 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT: 0x7 R_X86_64_8 foo 0x0
# CHECK-NEXT: 0x7 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT: 0x4 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT: 0x1 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT: 0x8 R_X86_64_8 bar 0x0
# CHECK-NEXT: 0x8 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT: 0x5 R_X86_64_NONE .Lbase 0x0
# CHECK-NEXT: 0x2 R_X86_64_NONE .Lbase 0x0

# CHECK:      0x4 R_X86_64_32 foo 0x0
# CHECK-NEXT: 0x4 R_X86_64_64 bar 0x0
# CHECK-NEXT: 0x4 R_X86_64_64 bar 0x0

.text
.Lbase:
  .byte 0, 0, 0, 0, 0, 0, 0, foo, bar

  .reloc .Lbase+6, R_X86_64_NONE, .Lbase
  .reloc .Lbase+3, R_X86_64_NONE, .Lbase
  .reloc .Lbase+0, R_X86_64_NONE, .Lbase

  .reloc .Lbase+7, R_X86_64_NONE, .Lbase
  .reloc .Lbase+4, R_X86_64_NONE, .Lbase
  .reloc .Lbase+1, R_X86_64_NONE, .Lbase

  .reloc .Lbase+8, R_X86_64_NONE, .Lbase
  .reloc .Lbase+5, R_X86_64_NONE, .Lbase
  .reloc .Lbase+2, R_X86_64_NONE, .Lbase

.data
  .long 0
  .reloc ., BFD_RELOC_64, bar
  .long foo
  .reloc .-4, BFD_RELOC_64, bar
