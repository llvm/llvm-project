; RUN: opt < %s -passes=instcombine -data-layout="p:32:32" -S | FileCheck %s
; RUN: opt < %s -passes=instcombine -data-layout="p:64:64" -S | FileCheck %s

define signext i32 @b(ptr inreg %x)   {
  ret i32 0
}

define void @c(...) {
  ret void
}

declare void @useit(i32)

define void @d(i32 %x, ...) {
  call void @useit(i32 %x)
  ret void
}

define void @naked_func() naked {
  tail call void asm sideeffect "mov  r1, r0", ""()
  unreachable
}

define void @g(ptr %y) {
  call i32 @b(i32 zeroext 0)
  call void @c(ptr %y)
  call void @c(ptr sret(i32) %y)
  call void @d(i32 0, ptr sret(i32) %y)
  call void @d(i32 0, ptr captures(none) %y)
  call void @d(ptr noundef captures(none) %y)
  call void @naked_func(i32 1)
  ret void
}
; CHECK-LABEL: define void @g(ptr %y)
; CHECK:    call i32 @b(i32 zeroext 0)
; CHECK:    call void @c(ptr %y)
; CHECK:    call void @c(ptr sret(i32) %y)
; CHECK:    call void @d(i32 0, ptr sret(i32) %y)
; CHECK:    call void @d(i32 0, ptr captures(none) %y)
; CHECK:    call void @d(ptr noundef captures(none) %y)
; CHECK:    call void @naked_func(i32 1)
