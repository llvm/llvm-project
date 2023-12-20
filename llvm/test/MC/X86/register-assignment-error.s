// RUN: not llvm-mc -triple x86_64 %s -o /dev/null 2>&1 | FileCheck %s

var_xdata = %rcx

// This used to crash.
.if var_xdata == 1
.endif
// CHECK: error: expected absolute expression