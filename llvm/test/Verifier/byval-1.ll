; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s

; CHECK: Attribute 'byval(i32)' applied to incompatible type!
; CHECK-NEXT: ptr @h
declare void @h(i32 byval(i32) %num)
