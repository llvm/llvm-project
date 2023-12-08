; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: expected type
declare void @c(swifterror ptr %a)
