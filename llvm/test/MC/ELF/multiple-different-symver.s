// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t 2>&1 | FileCheck %s

// CHECK: Multiple symbol versions defined for foo

.symver foo, foo@1
.symver foo, foo@2
