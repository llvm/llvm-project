; RUN: opt < %s -passes=loop-vectorize -S

define void @foo() local_unnamed_addr {
entry:
  %exitcond = icmp eq i64 3, 3
  br label %for.body

for.body:                                         ; preds = %entry
  %i.05 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %total1 = add nsw i64 %i.05, 3
  %inc = add nuw nsw i64 %i.05, 1
  br i1 %exitcond, label %for.end, label %for.body, !llvm.loop !0

for.end:                                          ; preds = %for.body
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.enable", i1 true}
