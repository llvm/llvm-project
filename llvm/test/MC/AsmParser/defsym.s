# RUN: llvm-mc -filetype=obj -triple=i386-unknown-elf -defsym a=7 -defsym b=11 %s | llvm-objdump -t - | FileCheck %s

.ifndef a
.err 
.endif

.if a<>7
.err
.endif

.ifndef b
.err
.endif

.if b<>11
.err
.endif

# CHECK: 00000007 l       *ABS*  00000000 a
# CHECK: 0000000b l       *ABS*  00000000 b
