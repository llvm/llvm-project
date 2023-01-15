; RUN: opt -passes=deadargelim -S < %s | FileCheck %s

define void @test(i32) {
  ret void
}

define void @foo() {
  call void @test(i32 0)
  ret void
; CHECK-LABEL: @foo(
; CHECK: i32 poison
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
; CHECK: tail call void @f(i32 poison)
  tail call void @f(i32 %add)
  ret void
}

define void @h() {
entry:
  %i = alloca i32, align 4
  store volatile i32 10, ptr %i, align 4
; CHECK: %tmp = load volatile i32, ptr %i, align 4
; CHECK-NEXT: call void @f(i32 poison)
  %tmp = load volatile i32, ptr %i, align 4
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

%swift_error = type opaque

define void @unused_swifterror_arg(ptr swifterror %dead_arg) {
  tail call void @sideeffect() nounwind
  ret void
}

; CHECK-LABEL: @dont_replace_by_poison
; CHECK-NOT: call void @unused_swifterror_arg({{.*}}poison)
define void @dont_replace_by_poison() {
  %error_ptr_ref = alloca swifterror ptr
  store ptr null, ptr %error_ptr_ref
  call void @unused_swifterror_arg(ptr %error_ptr_ref)
  ret void
}
