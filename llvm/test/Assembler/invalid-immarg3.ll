; RUN: not llvm-as < %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: error: this attribute does not apply to return values
declare immarg i32 @llvm.immarg.retattr(i32)
