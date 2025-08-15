# RUN: not llvm-mc -filetype=obj -triple=mips64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

# CHECK: :[[@LINE+1]]:11: error: cyclic dependency detected for symbol 'A'
.set A, A + 1
# CHECK: :[[@LINE+1]]:7: error: expected relocatable expression
.word A
.word A

# CHECK: :[[@LINE+2]]:11: error: cyclic dependency detected for symbol 'B'
# CHECK: :[[@LINE+1]]:11: error: expression could not be evaluated
.set B, B + 1
