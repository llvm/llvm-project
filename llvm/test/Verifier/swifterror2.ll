; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: this attribute does not apply to return values
declare swifterror void @c(ptr swifterror %a)
