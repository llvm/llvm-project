; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: invalid use of parameter-only attribute
declare swifterror void @c(i32** swifterror %a)
