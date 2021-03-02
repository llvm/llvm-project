; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(i32* swiftasync %a, i32* swiftasync %b)
; CHECK: Cannot have multiple 'swiftasync' parameters!
