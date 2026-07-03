# RUN: llvm-mc -triple=hexagon %s | FileCheck --check-prefix=PRINT %s
# RUN: llvm-mc -triple=hexagon -filetype=obj %s | llvm-objdump -r - | FileCheck %s

# PRINT: .reloc {{.*}}+8, R_HEX_NONE, .data
# PRINT: .reloc {{.*}}+4, R_HEX_NONE, foo+4
# PRINT: .reloc {{.*}}+0, R_HEX_NONE, 8
# PRINT: .reloc {{.*}}+0, R_HEX_16, .data+2
# PRINT: .reloc {{.*}}+0, R_HEX_32, foo+3
# PRINT: .reloc {{.*}}+0, BFD_RELOC_NONE, 9
# PRINT: .reloc {{.*}}+0, BFD_RELOC_16, 9
# PRINT: .reloc {{.*}}+0, BFD_RELOC_32, 9
.text
  .reloc .+8, R_HEX_NONE, .data
  .reloc .+4, R_HEX_NONE, foo+4
  .reloc .+0, R_HEX_NONE, 8

  .reloc .+0, R_HEX_8, .data+2
  .reloc .+0, R_HEX_16, .data+2
  .reloc .+0, R_HEX_32, foo+3

  .reloc .+0, BFD_RELOC_NONE, 9
  .reloc .+0, BFD_RELOC_8, 9
  .reloc .+0, BFD_RELOC_16, 9
  .reloc .+0, BFD_RELOC_32, 9
  nop
  nop
  nop

.data
.globl foo
foo:
  .word 0
  .word 0
  .word 0

# CHECK:      {{0+}}[[#%x,8]] R_HEX_NONE .data
# CHECK-NEXT: {{0+}}[[#%x,4]] R_HEX_NONE foo+0x4
# CHECK-NEXT: {{0+}} R_HEX_NONE *ABS*+0x8
# CHECK-NEXT: {{0+}} R_HEX_8 .data+0x2
# CHECK-NEXT: {{0+}} R_HEX_16 .data+0x2
# CHECK-NEXT: {{0+}} R_HEX_32 foo+0x3
# CHECK-NEXT: {{0+}} R_HEX_NONE *ABS*+0x9
# CHECK-NEXT: {{0+}} R_HEX_8 *ABS*+0x9
# CHECK-NEXT: {{0+}} R_HEX_16 *ABS*+0x9
# CHECK-NEXT: {{0+}} R_HEX_32 *ABS*+0x9
