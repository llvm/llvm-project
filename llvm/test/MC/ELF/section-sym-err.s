# RUN: not llvm-mc -filetype=obj -triple x86_64 %s -o %t 2>&1 | FileCheck %s

.section foo
foo:
# CHECK: [[#@LINE-1]]:1: error: symbol 'foo' is already defined

x1:
.section x1
# CHECK: <unknown>:0: error: invalid symbol redefinition
