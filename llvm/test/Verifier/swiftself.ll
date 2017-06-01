; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(i32* swiftself %a, i32* swiftself %b)
; CHECK: Cannot have multiple 'swiftself' parameters!
