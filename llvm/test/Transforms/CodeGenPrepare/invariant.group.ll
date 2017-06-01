; RUN: opt -codegenprepare -S < %s | FileCheck %s

@tmp = global i8 0

; CHECK-LABEL: define void @foo() {
define void @foo() {
enter:
  ; CHECK-NOT: !invariant.group
  ; CHECK-NOT: @llvm.invariant.group.barrier(
  ; CHECK: %val = load i8, i8* @tmp, !tbaa
  %val = load i8, i8* @tmp, !invariant.group !0, !tbaa !{!1, !1, i64 0}
  %ptr = call i8* @llvm.invariant.group.barrier(i8* @tmp)
  
  ; CHECK: store i8 42, i8* @tmp
  store i8 42, i8* %ptr, !invariant.group !0
  
  ret void
}
; CHECK-LABEL: }

declare i8* @llvm.invariant.group.barrier(i8*)

!0 = !{!"something"}
!1 = !{!"x", !0}
