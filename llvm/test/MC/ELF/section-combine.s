## Sections are differentiated by the quadruple
## (section_name, group_name, unique_id, link_to_symbol_name).
## Sections sharing the same quadruple are combined into one section.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-readelf -x .foo %t | FileCheck %s

# CHECK:      Hex dump of section '.foo':
# CHECK-NEXT: 0x00000000 00
# CHECK:      Hex dump of section '.foo':
# CHECK-NEXT: 0x00000000 0102
# CHECK:      Hex dump of section '.foo':
# CHECK-NEXT: 0x00000000 03
# CHECK:      Hex dump of section '.foo':
# CHECK-NEXT: 0x00000000 0405
# CHECK:      Hex dump of section '.foo':
# CHECK-NEXT: 0x00000000 06
# CHECK:      Hex dump of section '.foo':
# CHECK-NEXT: 0x00000000 0708

foo:
bar:

## foo and bar are in the same section. However, a section referencing foo
## is considered different from a section referencing bar.
.section .foo,"o",@progbits,foo
.byte 0

.section .foo,"o",@progbits,bar
.byte 1
.section .foo,"o",@progbits,bar
.byte 2

.section .foo,"o",@progbits,bar,unique,0
.byte 3

.section .foo,"o",@progbits,bar,unique,1
.byte 4
.section .foo,"o",@progbits,bar,unique,1
.byte 5

.section .foo,"Go",@progbits,comdat0,comdat,bar,unique,1
.byte 6

.section .foo,"Go",@progbits,comdat1,comdat,bar,unique,1
.byte 7
.section .foo,"Go",@progbits,comdat1,comdat,bar,unique,1
.byte 8
