; RUN: opt < %s -passes=loop-vectorize -S | FileCheck %s
; RUN: opt < %s -passes=loop-vectorize -prefer-predicate-over-epilogue=predicate-dont-vectorize -S | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; The test predication_in_loop corresponds
; to the following function
;   for (long long i = 0; i < 1024; i++) {
;    if (cond[i])
;      a[i] /= b[i];
;  }

define void  @predication_in_loop(ptr %a, ptr %b, ptr %cond) #0 {
; CHECK-LABEL: @predication_in_loop
; CHECK:  sdiv <vscale x 4 x i32>
;
entry:
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %i.09 = phi i64 [ %inc, %for.inc ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %cond, i64 %i.09
  %0 = load i32, ptr %arrayidx, align 4
  %tobool.not = icmp eq i32 %0, 0
  br i1 %tobool.not, label %for.inc, label %if.then

if.then:                                          ; preds = %for.body
  %arrayidx1 = getelementptr inbounds i32, ptr %b, i64 %i.09
  %1 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %i.09
  %2 = load i32, ptr %arrayidx2, align 4
  %div = sdiv i32 %2, %1
  store i32 %div, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.then
  %inc = add nuw nsw i64 %i.09, 1
  %exitcond.not = icmp eq i64 %inc, 1024
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !0
}


;
; This test unpredicated_loop_predication_through_tailfolding corresponds
; to the following function
;   for (long long i = 0; i < 1024; i++) {
;      a[i] /= b[i];
;  }

; Scalarization not possible in the main loop when there is no predication and
; epilogue should not be able to allow scalarization
; otherwise it could  be able to vectorize, but will not because
; "Max legal vector width too small, scalable vectorization unfeasible.."

define void @unpredicated_loop_predication_through_tailfolding(ptr %a, ptr %b) #0 {
; CHECK-LABEL: @unpredicated_loop_predication_through_tailfolding
; CHECK-NOT:  sdiv <vscale x 4 x i32>

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %b, i64 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %sdiv = sdiv i32 %1, %0
  %2 = add nuw nsw i64 %iv, 8
  %arrayidx5 = getelementptr inbounds i32, ptr %a, i64 %2
  store i32 %sdiv, ptr %arrayidx5, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, 1024
  br i1 %exitcond.not, label %exit, label %loop, !llvm.loop !0

exit:
  ret void

}

attributes #0 = { "target-features"="+sve" }

!0 = distinct !{!0, !1, !2, !3, !4}
!1 = !{!"llvm.loop.vectorize.width", i32 4}
!2 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
!3 = !{!"llvm.loop.interleave.count", i32 1}
!4 = !{!"llvm.loop.vectorize.enable", i1 true}
