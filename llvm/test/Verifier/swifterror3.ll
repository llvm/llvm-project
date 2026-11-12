; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

; CHECK: expected type
declare void @c(swifterror ptr %a)
