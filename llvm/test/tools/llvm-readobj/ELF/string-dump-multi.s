# REQUIRES: x86-registered-target

# RUN: llvm-mc -filetype=obj -triple x86_64 %s -o %t.o
# RUN: llvm-readobj -p .a -p .b %t.o | FileCheck %s
# RUN: llvm-readelf -p .a -p .b %t.o | FileCheck %s

# CHECK:      String dump of section '.a':
# CHECK-NEXT: [     0] 0
# CHECK-EMPTY:
# CHECK-NEXT: String dump of section '.b':
# CHECK-NEXT: [     0] 1
# CHECK-EMPTY:
# CHECK-NEXT: String dump of section '.a':
# CHECK-NEXT: [     0] 2

.section .a,"a",@progbits,unique,0
.asciz "0"
.section .b,"a",@progbits
.asciz "1"
.section .a,"a",@progbits,unique,1
.asciz "2"
