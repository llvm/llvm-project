; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(i32** swifterror %a, i32** swifterror %b)
; CHECK: Cannot have multiple 'swifterror' parameters!
