; RUN: opt -deadargelim -S < %s | FileCheck %s

define void @test(i32) {
  ret void
}

define void @foo() {
  call void @test(i32 0)
  ret void
; CHECK-LABEL: @foo(
; CHECK: i32 undef
}

define void @f(i32 %X) {
entry:
  tail call void @sideeffect() nounwind
  ret void
}

declare void @sideeffect()

define void @g(i32 %n) {
entry:
  %add = add nsw i32 %n, 1
; CHECK: tail call void @f(i32 undef)
  tail call void @f(i32 %add)
  ret void
}

define void @h() {
entry:
  %i = alloca i32, align 4
  store volatile i32 10, i32* %i, align 4
; CHECK: %tmp = load volatile i32, i32* %i, align 4
; CHECK-NEXT: call void @f(i32 undef)
  %tmp = load volatile i32, i32* %i, align 4
  call void @f(i32 %tmp)
  ret void
}

; Check that callers are not transformed for weak definitions.
define weak i32 @weak_f(i32 %x) nounwind {
entry:
  ret i32 0
}
define void @weak_f_caller() nounwind {
entry:
; CHECK: call i32 @weak_f(i32 10)
  %call = tail call i32 @weak_f(i32 10)
  ret void
}

