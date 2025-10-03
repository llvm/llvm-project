# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s

# CHECK: [[#@LINE+2]]:7: error: cyclic dependency detected for symbol 'a'
# CHECK: [[#@LINE+1]]:7: error: expression could not be evaluated
a = a + 1

# CHECK: [[#@LINE+3]]:6: error: cyclic dependency detected for symbol 'b1'
# CHECK: [[#@LINE+1]]:6: error: expression could not be evaluated
b0 = b1
b1 = b2
b2 = b0

# CHECK: [[#@LINE+3]]:6: error: cyclic dependency detected for symbol 'c1'
# CHECK: [[#@LINE+1]]:9: error: expression could not be evaluated
c0 = c1 + 1
c1 = c0
