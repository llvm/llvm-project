; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s

; CHECK: error: this attribute does not apply to parameters
declare void @foo(i32 safestack %x)
