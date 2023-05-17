// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t
// RUN: llvm-jitlink -noexec %t

.section	.foo,"axG",@progbits,g1,comdat
.globl  g1
g1:
call test1
retq

.section	.baz,"axG",@progbits,g1,comdat
test1:
retq

.section	.bar,"axG",@progbits,g2,comdat
.globl  g2
g2:
call test2
retq

.section	.baz,"axG",@progbits,g2,comdat
test2:
retq

.text
.globl  main
main:
retq
