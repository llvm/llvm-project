// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o - 2>&1 | FileCheck %s

// CHECK-NOT: Multiple symbol versions defined for foo

.symver foo, foo@1
.symver foo, foo@1
