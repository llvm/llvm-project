; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @a(i32* sret(i32) %a, i32* sret(i32) %b)
; CHECK: Cannot have multiple 'sret' parameters!

declare void @b(i32* %a, i32* %b, i32* sret(i32) %c)
; CHECK: Attribute 'sret' is not on first or second parameter!

; CHECK: Attribute 'sret(i32)' applied to incompatible type!
; CHECK-NEXT: void (i32)* @not_ptr
declare void @not_ptr(i32 sret(i32) %x)
