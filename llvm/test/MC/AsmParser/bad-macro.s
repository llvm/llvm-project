// RUN: not llvm-mc -triple x86_64-apple-darwin10 %s 2>&1 | FileCheck %s

.macro 23

// CHECK: expected identifier in '.macro' directive

.macro abc 33

// CHECK: expected identifier in '.macro' directive
