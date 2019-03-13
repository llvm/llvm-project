; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: invalid use of parameter-only attribute
declare immarg i32 @llvm.immarg.retattr(i32)
