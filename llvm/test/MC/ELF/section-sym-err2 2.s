// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o 2>&1 | FileCheck %s

foo:
.section foo

// CHECK: error: invalid symbol redefinition
