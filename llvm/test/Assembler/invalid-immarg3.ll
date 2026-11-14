; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: error: this attribute does not apply to return values
declare immarg i32 @llvm.immarg.retattr(i32)
