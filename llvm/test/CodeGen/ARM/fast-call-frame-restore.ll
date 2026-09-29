; RUN: llc -mtriple=armv7-linux-gnueabi %s -o - | FileCheck %s

declare void @bar(ptr, i32, i32, i32, i32)

define void @foo(i32 %amt) optnone noinline {
  br label %next

next:
  %mem = alloca i8;, i32 %amt
  br label %next1

next1:
  call void @bar(ptr %mem, i32 undef, i32 undef, i32 undef, i32 undef)
; CHECK: sub sp, sp, #8
; CHECK: bl bar
; CHECK: add sp, sp, #8

  ret void
}
