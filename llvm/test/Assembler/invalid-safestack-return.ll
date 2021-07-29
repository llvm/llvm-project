; RUN: not llvm-as -o /dev/null %s 2>&1 | FileCheck %s

; CHECK: error: this attribute does not apply to return values
declare safestack void @foo()
