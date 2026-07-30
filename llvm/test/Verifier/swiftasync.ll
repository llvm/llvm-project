; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(ptr swiftasync %a, ptr swiftasync %b)
; CHECK: Cannot have multiple 'swiftasync' parameters!
