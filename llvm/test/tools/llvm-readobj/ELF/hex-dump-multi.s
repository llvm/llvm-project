# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t.o
# RUN: llvm-readobj -x .a -x .b %t.o | FileCheck %s
# RUN: llvm-readelf -x .a -x .b %t.o | FileCheck %s

# CHECK:      Hex dump of section '.a':
# CHECK-NEXT: 0x00000000 00
# CHECK-EMPTY:
# CHECK-NEXT: Hex dump of section '.b':
# CHECK-NEXT: 0x00000000 01
# CHECK-EMPTY:
# CHECK-NEXT: Hex dump of section '.a':
# CHECK-NEXT: 0x00000000 02

.section .a,"a",@progbits,unique,0
.byte 0
.section .b,"a",@progbits
.byte 1
.section .a,"a",@progbits,unique,1
.byte 2
