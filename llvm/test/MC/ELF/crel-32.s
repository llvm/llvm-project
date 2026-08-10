# REQUIRES: arm-registered-target
## Test CREL for a 32-bit big-endian target.
## CREL enables the RELA form even if ARM normally uses REL.

# RUN: llvm-mc -filetype=obj -crel -triple=armv8 %s -o %t.o
# RUN: llvm-readelf -Sr %t.o | FileCheck %s

# CHECK:      [ 3] .data           PROGBITS      00000000 {{.*}} 000008 00  WA  0   0  1
# CHECK-NEXT: [ 4] .crel.data      CREL          00000000 {{.*}} 00000a 01   I  5   3  1

# CHECK:      Relocation section '.crel.data' at offset {{.*}} contains 2 entries:
# CHECK-NEXT:  Offset     Info    Type                Sym. Value  Symbol's Name + Addend
# CHECK-NEXT: 00000000  {{.*}}   R_ARM_NONE             00000000   a0 - 1
# CHECK-NEXT: 00000004  {{.*}}   R_ARM_ABS32            00000000   a3 + 4000

.data
.reloc .+0, BFD_RELOC_NONE, a0-1
.reloc .+4, BFD_RELOC_32, a3+0x4000
.quad 0
