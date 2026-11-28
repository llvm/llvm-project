; RUN: not llvm-as -disable-output %s 2>&1 | FileCheck %s

; CHECK: error: this attribute does not apply to return values
declare safestack void @foo()
