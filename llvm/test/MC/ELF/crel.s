# RUN: llvm-mc -filetype=obj -crel -triple=x86_64 %s -o %t.o
# RUN: llvm-readelf -Sr -x .crelrodata2 -x .crelrodata16 %t.o | FileCheck %s

# RUN: %if aarch64-registered-target %{ llvm-mc -filetype=obj -crel -triple=aarch64_be %s -o %t.be.o %}
# RUN: %if aarch64-registered-target %{ llvm-readelf -r %t.be.o | FileCheck %s --check-prefix=A64BE %}

# CHECK:      [ 4] .data             PROGBITS      0000000000000000 {{.*}} 000008 00  WA  0   0  1
# CHECK-NEXT: [ 5] .crel.data        CREL          0000000000000000 {{.*}} 00002c 01   I 14   4  1
# CHECK-NEXT: [ 6] .rodata           PROGBITS      0000000000000000 {{.*}} 00002b 00   A  0   0  1
# CHECK-NEXT: [ 7] .crel.rodata      CREL          0000000000000000 {{.*}} 000010 01   I 14   6  1
# CHECK-NEXT: [ 8] rodata2           PROGBITS      0000000000000000 {{.*}} 000008 00   A  0   0  1
# CHECK-NEXT: [ 9] .crelrodata2      CREL          0000000000000000 {{.*}} 000005 01   I 14   8  1
# CHECK-NEXT: [10] rodata16          PROGBITS      0000000000000000 {{.*}} 000010 00   A  0   0  1
# CHECK-NEXT: [11] .crelrodata16     CREL          0000000000000000 {{.*}} 000004 01   I 14  10  1
# CHECK-NEXT: [12] noalloc           PROGBITS      0000000000000000 {{.*}} 000004 00      0   0  1
# CHECK-NEXT: [13] .crelnoalloc      CREL          0000000000000000 {{.*}} 000005 01   I 14  12  1
# CHECK-NEXT: [14] .symtab           SYMTAB

# CHECK:      Relocation section '.crel.data' at offset {{.*}} contains 8 entries:
# CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT: 0000000000000000  {{.*}}           R_X86_64_NONE          0000000000000000 a0 + 0
# CHECK-NEXT: 0000000000000001  {{.*}}           R_X86_64_NONE          0000000000000000 a1 - 1
# CHECK-NEXT: 0000000000000002  {{.*}}           R_X86_64_NONE          0000000000000000 a2 - 1
# CHECK-NEXT: 0000000000000004  {{.*}}           R_X86_64_32            0000000000000000 a3 + 4000
# CHECK-NEXT: 0000000000000005  {{.*}}           R_X86_64_64            0000000000000000 a3 - 8000000000000000
# CHECK-NEXT: 0000000000000005  {{.*}}           R_X86_64_64            0000000000000000 a1 + 7fffffffffffffff
# CHECK-NEXT: 0000000000000005  {{.*}}           R_X86_64_32            0000000000000000 a1 - 1
# CHECK-NEXT: 0000000000000005  {{.*}}           R_X86_64_64            0000000000000000 a2 - 1
# CHECK-EMPTY:
# CHECK-NEXT: Relocation section '.crel.rodata' at offset {{.*}} contains 4 entries:
# CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT: 0000000000000000  {{.*}}           R_X86_64_PC32          0000000000000000 foo + 0
# CHECK-NEXT: 000000000000000f  {{.*}}           R_X86_64_PC32          0000000000000000 foo + 3f
# CHECK-NEXT: 000000000000001f  {{.*}}           R_X86_64_PC64          0000000000000000 foo + 7f
# CHECK-NEXT: 0000000000000027  {{.*}}           R_X86_64_PC32          0000000000000000 _start - 1f81
# CHECK-EMPTY:
# CHECK-NEXT: Relocation section '.crelrodata2' at offset {{.*}} contains 2 entries:
# CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT: 0000000000000002  {{.*}}           R_X86_64_32            0000000000000000 .data + 0
# CHECK-NEXT: 0000000000000008  {{.*}}           R_X86_64_32            0000000000000000 .data + 0
# CHECK-EMPTY:
# CHECK-NEXT: Relocation section '.crelrodata16' at offset {{.*}} contains 1 entries:
# CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT: 0000000000000008  {{.*}}           R_X86_64_64            0000000000000000 rodata16 + 0
# CHECK-EMPTY:
# CHECK-NEXT: Relocation section '.crelnoalloc' at offset {{.*}} contains 1 entries:
# CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT: 0000000000000000  {{.*}}           R_X86_64_32            0000000000000000 .text + 4

## count * 8 + 4 + shift = 2*8+4+1 = 21
# CHECK:      Hex dump of section '.crelrodata2':
# CHECK-NEXT: 0x00000000 150b020a 18                         .
## count * 8 + 4 + shift = 1*8+4+3 = 15
# CHECK:      Hex dump of section '.crelrodata16':
# CHECK-NEXT: 0x00000000 0f0b0301 .

# A64BE:      0000000000000000  {{.*}}           R_AARCH64_NONE         0000000000000000 a0 + 0
# A64BE-NEXT: 0000000000000001  {{.*}}           R_AARCH64_NONE         0000000000000000 a1 - 1
# A64BE-NEXT: 0000000000000002  {{.*}}           R_AARCH64_NONE         0000000000000000 a2 - 1
# A64BE-NEXT: 0000000000000004  {{.*}}           R_AARCH64_ABS32        0000000000000000 a3 + 4000
# A64BE-NEXT: 0000000000000005  {{.*}}           R_AARCH64_ABS64        0000000000000000 a3 - 8000000000000000
# A64BE-NEXT: 0000000000000005  {{.*}}           R_AARCH64_ABS64        0000000000000000 a1 + 7fffffffffffffff
# A64BE-NEXT: 0000000000000005  {{.*}}           R_AARCH64_ABS32        0000000000000000 a1 - 1
# A64BE-NEXT: 0000000000000005  {{.*}}           R_AARCH64_ABS64        0000000000000000 a2 - 1
# A64BE-EMPTY:

.globl _start
_start:
  ret

.section .text.1,"ax"
  ret

## Test various combinations of delta offset and flags, delta symbol index
## (if present), delta type (if present), delta addend (if present).
.data
.reloc .+0, BFD_RELOC_NONE, a0
.reloc .+1, BFD_RELOC_NONE, a1-1  // same type
.reloc .+2, BFD_RELOC_NONE, a2-1  // same type and addend
.reloc .+4, BFD_RELOC_32, a3+0x4000
.reloc .+5, BFD_RELOC_64, a3-0x8000000000000000  // same symbol index
.reloc .+5, BFD_RELOC_64, a1+0x7fffffffffffffff  // same offset and type; addend+=UINT64_MAX
.reloc .+5, BFD_RELOC_32, a1-1  // same offset and symbol index
.reloc .+5, BFD_RELOC_64, a2-1  // same offset and addend
.quad 0

.section .rodata,"a"
.long foo - .
.space 15-4
.long foo - . + 63  // offset+=15
.space 16-4
.quad foo - . + 127  // offset+=16
.long _start - . - 8065

.section rodata2,"a"
.space 2
.reloc ., BFD_RELOC_32, .data
.space 6
.reloc ., BFD_RELOC_32, .data

.section rodata16,"a"
.quad 0
.quad rodata16

.section noalloc
.long .text + 4
