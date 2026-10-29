; RUN: opt -passes='cgscc(inline,function(instcombine))' -S < %s | FileCheck %s

; Check that devirtualization of an indirect call to itself doesn't cause
; infinite inlining. Note that we need the devirtualization to happen in
; simplification between inlining runs or else InlineCost will substitute %p
; with @p and see that the call is a recursive call and bail.

define void @p(ptr %p, i64 %x) {
  %a = alloca ptr
  store ptr %p, ptr %a
  %g = getelementptr i8, ptr %a, i64 %x
  %b = load ptr, ptr %g
  call void %b(ptr %p, i64 %x)
  ret void
}

define void @q() {
; CHECK-LABEL: define void @q() {
; CHECK-NEXT:    call void @p({{.*}}), !inline_history [[HISTORY:![0-9]+]]
; CHECK-NEXT:    ret void
  call void @p(ptr @p, i64 0)
  ret void
}

; CHECK: [[HISTORY]] = !{ptr @p}
