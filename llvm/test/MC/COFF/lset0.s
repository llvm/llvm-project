// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | llvm-nm - | FileCheck %s --check-prefix=GLOBAL
// RUN: llvm-mc -filetype=obj -triple i686-pc-win32 %s -o - | llvm-nm - | FileCheck %s --check-prefix=LOCAL

not_global = 123
global = 456
.globl global
Llocal = 789

// LOCAL-NOT: local
// GLOBAL: A global
// GLOBAL: a not_global
