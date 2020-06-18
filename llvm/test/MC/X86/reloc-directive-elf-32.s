# RUN: llvm-mc -triple=i386-pc-linux-musl %s | FileCheck --check-prefix=PRINT %s

# RUN: llvm-mc -filetype=obj -triple=i386-pc-linux-musl %s -o %t
# RUN: llvm-readobj -r %t | FileCheck %s
# RUN: llvm-readelf -x .data %t | FileCheck --check-prefix=HEX %s

# PRINT:      .reloc 2, R_386_NONE, .data
# PRINT-NEXT: .reloc 1, R_386_NONE, foo+4
# PRINT-NEXT: .reloc 0, R_386_NONE, 8
# PRINT-NEXT: .reloc 0, R_386_32, .data+2
# PRINT-NEXT: .reloc 0, R_386_IRELATIVE, foo+3
# PRINT-NEXT: .reloc 0, R_386_GOT32X, 5

# X86 relocations use the Elf32_Rel format. Addends are neither stored in the
# relocation entries nor applied in the referenced locations.
# CHECK:      0x2 R_386_NONE .data 0x0
# CHECK-NEXT: 0x1 R_386_NONE foo 0x0
# CHECK-NEXT: 0x0 R_386_NONE - 0x0
# CHECK-NEXT: 0x0 R_386_32 .data 0x0
# CHECK-NEXT: 0x0 R_386_IRELATIVE foo 0x0
# CHECK-NEXT: 0x0 R_386_GOT32X - 0x0

# HEX: 0x00000000 00000000 00000000

.text
  ret
  nop
  nop
  .reloc 2, R_386_NONE, .data
  .reloc 1, R_386_NONE, foo+4
  .reloc 0, R_386_NONE, 8
  .reloc 0, R_386_32, .data+2
  .reloc 0, R_386_IRELATIVE, foo+3
  .reloc 0, R_386_GOT32X, 5

.data
.globl foo
foo:
  .long 0
  .long 0
