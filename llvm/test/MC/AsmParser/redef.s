# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t
# RUN: llvm-objdump -ts %t | FileCheck %s

# CHECK:      SYMBOL TABLE:
# CHECK-NEXT: 0000000000000000 l       .text  0000000000000000 l
# CHECK-NEXT: 0000000000000008 l       *ABS*  0000000000000000 x
# CHECK-NEXT: 0000000000000002 l       *ABS*  0000000000000000 b
# CHECK-NEXT: 0000000000000003 l       *ABS*  0000000000000000 c
# CHECK-NEXT: ffffffffffffffff l       *ABS*  0000000000000000 a
# CHECK-NEXT: 0000000000000000 g       .text  0000000000000000 l_v
# CHECK:      Contents of section .data:
# CHECK-NEXT:  0000 00000000 04000000 08000000           .
# CHECK-NEXT: Contents of section .data1:
# CHECK-NEXT:  0000 010203ff 0203                        ......

l:

.data
.set x, 0
.long x
x = .-.data
.long x
.set x,.-.data
.long x

.globl l_v
.set l_v, l
.globl l_v
.set l_v, l

.section .data1,"aw"
.equiv b, 2*a
.set a, 1
.equiv c, 3*a

.if b > a
.byte a, b, c
.endif

.set a, -a
.if b > a
.byte a, b, c
.endif
