; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

declare void @a(ptr swiftself %a, ptr swiftself %b)
; CHECK: Cannot have multiple 'swiftself' parameters!
