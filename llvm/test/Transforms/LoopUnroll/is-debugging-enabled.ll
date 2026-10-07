; RUN: opt -passes=loop-unroll -S < %s | FileCheck %s

; Full unrolling may duplicate the static query site because it preserves the
; three dynamically executed observations from the original loop.
define void @query_allows_full_unrolling() {
; CHECK-LABEL: define void @query_allows_full_unrolling() {
; CHECK:       loop:
; CHECK-NEXT:    call i1 @llvm.is.debugging.enabled()
; CHECK-NEXT:    call i1 @llvm.is.debugging.enabled()
; CHECK-NEXT:    call i1 @llvm.is.debugging.enabled()
; CHECK-NEXT:    ret void
;
entry:
  br label %loop

loop:
  %index = phi i32 [ 0, %entry ], [ %next, %loop ]
  %enabled = call i1 @llvm.is.debugging.enabled()
  %next = add nuw nsw i32 %index, 1
  %done = icmp eq i32 %next, 3
  br i1 %done, label %exit, label %loop, !llvm.loop !0

exit:
  ret void
}

declare i1 @llvm.is.debugging.enabled()

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
