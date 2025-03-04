; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(ptr swiftcoro %a, ptr swiftcoro %b)
; CHECK: Cannot have multiple 'swiftcoro' parameters!
