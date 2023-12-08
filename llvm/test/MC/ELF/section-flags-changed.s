# RUN: not llvm-mc -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

foo:
.section .foo,"ax",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section flags for .foo, expected: 0x6
.section .foo,"awx",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section flags for .foo, expected: 0x6
.pushsection .foo,"a",@progbits

# CHECK: {{.*}}.s:[[# @LINE+1]]:1: error: changed section flags for .foo, expected: 0x6
.section .foo,"",@progbits

.pushsection .foo,"ax",@progbits
