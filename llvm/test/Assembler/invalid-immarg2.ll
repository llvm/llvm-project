; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: invalid use of parameter-only attribute on a function
declare void @llvm.immarg.func() immarg
