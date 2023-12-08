; RUN: llc < %s -mtriple=i386-apple-darwin

define fastcc void @foo(i32 %type) nounwind optsize {
entry:
  switch i32 %type, label %bb26 [
    i32 33634, label %bb11
    i32 5121, label %bb27
  ]

bb11:                                             ; preds = %entry
  br label %bb27

bb26:                                             ; preds = %entry
  unreachable

bb27:                                             ; preds = %bb11, %entry
  %srcpb.0 = phi i32 [ 1, %bb11 ], [ 0, %entry ]
  br i1 undef, label %bb348, label %bb30.lr.ph

bb30.lr.ph:                                       ; preds = %bb27
  %.sum743 = shl i32 %srcpb.0, 1
  %0 = mul i32 %srcpb.0, -2
  %.sum745 = add i32 %.sum743, %0
  br i1 undef, label %bb70, label %bb71

bb70:                                             ; preds = %bb30.lr.ph
  unreachable

bb71:                                             ; preds = %bb30.lr.ph
  br i1 undef, label %bb92, label %bb80

bb80:                                             ; preds = %bb71
  unreachable

bb92:                                             ; preds = %bb71
  %1 = getelementptr inbounds i8, ptr undef, i32 %.sum745
  unreachable

bb348:                                            ; preds = %bb27
  ret void
}
