# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: not llvm-mc -filetype=obj -triple=mips64 1.s -o /dev/null 2>&1 | FileCheck 1.s --implicit-check-not=error:
# RUN: not llvm-mc -filetype=obj -triple=mips64 2.s -o /dev/null 2>&1 | FileCheck 2.s --check-prefix=CHECK2 --implicit-check-not=error:

#--- 1.s
# CHECK: :[[@LINE+1]]:11: error: cyclic dependency detected for symbol 'A'
.set A, A + 1
# CHECK: :[[@LINE+1]]:7: error: expected relocatable expression
.word A
.word A

#--- 2.s
# CHECK2: :[[@LINE+2]]:11: error: cyclic dependency detected for symbol 'B'
# CHECK2: :[[@LINE+1]]:11: error: expression could not be evaluated
.set B, B + 1
