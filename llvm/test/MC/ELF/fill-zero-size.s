# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -x .data -s %t | FileCheck %s

## A .fill with a size of zero emits nothing, regardless of the repeat count.
## It used to crash with a division by zero when writing the fill fragment.

# CHECK:      0000000000000001 0 NOTYPE LOCAL DEFAULT [[#]] a
# CHECK-NEXT: 0000000000000002 0 NOTYPE LOCAL DEFAULT [[#]] b
# CHECK-NEXT: 0000000000000003 0 NOTYPE LOCAL DEFAULT [[#]] c
# CHECK-NEXT: 0000000000000004 0 NOTYPE LOCAL DEFAULT [[#]] d

# CHECK:      Hex dump of section '.data':
# CHECK-NEXT: 0x00000000 aabbccdd ee .....
# CHECK-NOT:  {{.}}

.data
.byte 0xaa
a:
.fill 6, 0
.byte 0xbb
b:
.fill 0, 0, 0x12
.byte 0xcc
c:
## Non-constant repeat count.
.fill 16 - (. - a), 0
.byte 0xdd
d:
.fill 3, 0, 0xffffffff
.byte 0xee

## No error for a non-zero fill value in a BSS section, since nothing is emitted.
.bss
.fill 3, 0, 0xff
