; RUN: opt -S %s -ipsccp | FileCheck %s

@a = internal global i32 2

define i32 @patatino() {
; CHECK: @patatino(
; CHECK: call void @f(i32 undef, i32 1)
; CHECK-NEXT: call void @f(i32 2, i32 0)
; CHECK-NEXT: ret i32 0
entry:
  call void @f(i32 undef, i32 1)
  %0 = load i32, i32* @a
  call void @f(i32 %0, i32 0)
  ret i32 0
}

define internal void @f(i32 %c, i32 %d) {
; CHECK: @f(
; CHECK:    ret void
;
entry:
  %cmp = icmp ne i32 %c, %d
  ret void
}
