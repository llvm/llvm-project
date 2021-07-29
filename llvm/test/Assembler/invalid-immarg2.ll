; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: this attribute does not apply to functions
declare void @llvm.immarg.func() immarg
