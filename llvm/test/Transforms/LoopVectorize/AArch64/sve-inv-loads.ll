; RUN: opt -S -passes=loop-vectorize -mattr=+sve -mtriple aarch64-linux-gnu \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue < %s | FileCheck %s

define void @invariant_load(i64 %n, ptr noalias nocapture %a, ptr nocapture readonly %b) {
; CHECK-LABEL: @invariant_load
; CHECK: vector.body:
; CHECK: %[[GEP:.*]] = getelementptr inbounds i32, ptr %b, i64 42
; CHECK-NEXT: %[[INVLOAD:.*]] = load i32, ptr %[[GEP]]
; CHECK-NEXT: %[[SPLATINS:.*]] = insertelement <vscale x 4 x i32> poison, i32 %[[INVLOAD]], i64 0
; CHECK-NEXT: %[[SPLAT:.*]] = shufflevector <vscale x 4 x i32> %[[SPLATINS]], <vscale x 4 x i32> poison, <vscale x 4 x i32> zeroinitializer
; CHECK: %[[LOAD:.*]] = load <vscale x 4 x i32>, ptr
; CHECK-NEXT: %[[ADD:.*]] = add nsw <vscale x 4 x i32> %[[SPLAT]], %[[LOAD]]
; CHECK: store <vscale x 4 x i32> %[[ADD]], ptr
entry:
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %b, i64 42
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds i32, ptr %b, i64 %iv
  %1 = load i32, ptr %arrayidx1, align 4
  %add = add nsw i32 %0, %1
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %iv
  store i32 %add, ptr %arrayidx2, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %n
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !1

for.end:                                          ; preds = %for.body
  ret void
}

!1 = distinct !{!1, !2, !3, !4}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.interleave.count", i32 1}
!4 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
