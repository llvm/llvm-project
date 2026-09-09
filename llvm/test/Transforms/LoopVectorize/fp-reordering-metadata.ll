; RUN: opt -passes='loop-vectorize' -force-vector-width=4 -S < %s 2>&1 | FileCheck %s

; Tests for llvm.loop.vectorize.fp_reordering metadata.

; Test 1: vectorize.enable only — default, FP reordering allowed.
; CHECK-LABEL: @fp_reduction_enable_only
; CHECK: vector.body
; CHECK: call float @llvm.vector.reduce.fadd.{{.*}}(float -0.000000e+00,
define float @fp_reduction_enable_only(ptr %A, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %sum.next, %loop ]
  %ptr = getelementptr inbounds float, ptr %A, i64 %i
  %val = load float, ptr %ptr, align 4
  %sum.next = fadd float %sum, %val
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %exit, label %loop, !llvm.loop !0

exit:
  ret float %sum.next
}

; Test 2: vectorize.enable + fp_reordering.enable — FP reordering explicitly allowed.
; CHECK-LABEL: @fp_reduction_enable_fp_reordering_true
; CHECK: vector.body
; CHECK: call float @llvm.vector.reduce.fadd.{{.*}}(float -0.000000e+00,
define float @fp_reduction_enable_fp_reordering_true(ptr %A, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %sum.next, %loop ]
  %ptr = getelementptr inbounds float, ptr %A, i64 %i
  %val = load float, ptr %ptr, align 4
  %sum.next = fadd float %sum, %val
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %exit, label %loop, !llvm.loop !2

exit:
  ret float %sum.next
}

; Test 3: vectorize.enable + fp_reordering.disable — FP reordering suppressed, loop not vectorized.
; CHECK-LABEL: @fp_reduction_enable_fp_reordering_false
; CHECK-NOT: vector.body
define float @fp_reduction_enable_fp_reordering_false(ptr %A, i64 %n) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %sum = phi float [ 0.0, %entry ], [ %sum.next, %loop ]
  %ptr = getelementptr inbounds float, ptr %A, i64 %i
  %val = load float, ptr %ptr, align 4
  %sum.next = fadd float %sum, %val
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, %n
  br i1 %cond, label %exit, label %loop, !llvm.loop !4

exit:
  ret float %sum.next
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable"}

!2 = distinct !{!2, !3, !5}
!3 = !{!"llvm.loop.vectorize.enable"}
!5 = !{!"llvm.loop.vectorize.fp_reordering.enable"}

!4 = distinct !{!4, !6, !7}
!6 = !{!"llvm.loop.vectorize.enable"}
!7 = !{!"llvm.loop.vectorize.fp_reordering.disable"}
