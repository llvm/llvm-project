; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s

declare void @a(ptr sret(i32) %a, ptr sret(i32) %b)
; CHECK: Cannot have multiple 'sret' parameters!

declare void @b(ptr %a, ptr %b, ptr sret(i32) %c)
; CHECK: Attribute 'sret' is not on first or second parameter!

; CHECK: Attribute 'sret(i32)' applied to incompatible type!
; CHECK-NEXT: ptr @not_ptr
declare void @not_ptr(i32 sret(i32) %x)
