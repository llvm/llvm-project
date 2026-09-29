# RUN: not llvm-mc -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

foo:
.section .foo,"aM",@progbits,1

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section entsize for .foo, expected: 1
.section .foo,"aM",@progbits,4

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section entsize for .foo, expected: 1
.pushsection .foo,"aM",@progbits,4

.pushsection .foo,"aM",@progbits,1
