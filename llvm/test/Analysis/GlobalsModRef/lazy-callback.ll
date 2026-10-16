; RUN: opt < %s -aa-pipeline=globals-aa -passes='require<globals-aa>,dse' -S | FileCheck %s
;
; Callback targets are modeled like legacy CallGraph edges: a modeled (not
; erased) declaration may invoke its callback, so the callback's effects must
; propagate to the caller.
;
; @broker is readnone (modeled as no effects) but may invoke @reader, which
; reads @CBX. The read in the callback observes the first store in @main_cb,
; so both stores must survive.
;
; @pure_broker is readnone and its callback @pure has no memory effects, so
; the first store in @main_pure must still be removed (no over-poisoning).
;
; The nounwind attributes are load-bearing: may-throw calls keep stores via
; unwind paths regardless of AA.

@CBX = internal global i32 0
@PBX = internal global i32 0

define void @reader() nounwind {
  %v = load i32, ptr @CBX
  call void @sink(i32 %v)
  ret void
}

define void @pure() nounwind {
  ret void
}

define void @user_cb() nounwind {
  call void (ptr, ptr) @broker(ptr null, ptr @reader)
  ret void
}

define void @main_cb() {
; CHECK-LABEL: define void @main_cb(
; CHECK: store i32 1, ptr @CBX
; CHECK-NEXT: call void @user_cb()
; CHECK-NEXT: store i32 2, ptr @CBX
  store i32 1, ptr @CBX
  call void @user_cb()
  store i32 2, ptr @CBX
  ret void
}

define void @user_pure() nounwind {
  call void @pure_broker(ptr @pure)
  ret void
}

define void @main_pure() {
; CHECK-LABEL: define void @main_pure(
; CHECK-NOT: store i32 1, ptr @PBX
; CHECK: call void @user_pure()
; CHECK-NEXT: store i32 2, ptr @PBX
  store i32 1, ptr @PBX
  call void @user_pure()
  store i32 2, ptr @PBX
  ret void
}

declare void @sink(i32) readnone nounwind
declare !callback !0 void @broker(ptr, ptr) readnone nounwind
declare !callback !2 void @pure_broker(ptr) readnone nounwind

!0 = !{!1}
!1 = !{i64 1, i64 -1}
!2 = !{!3}
!3 = !{i64 0, i64 -1}
