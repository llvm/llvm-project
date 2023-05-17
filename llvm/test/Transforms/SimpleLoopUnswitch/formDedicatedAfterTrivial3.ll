; RUN: opt < %s -passes=simple-loop-unswitch -disable-output

; PR38283
; PR38737
declare void @func_1()

define void @func_9(i32 signext %arg) {
bb:
  br label %bb5
bb5:                                              ; preds = %bb24, %bb
  %tmp3.0 = phi i32 [ undef, %bb ], [ %tmp29, %bb24 ]
  %tmp11 = icmp eq i32 %arg, 0
  %tmp15 = icmp eq i32 %tmp3.0, 0
  %spec.select = select i1 %tmp15, i32 0, i32 49
  %tmp1.2 = select i1 %tmp11, i32 %spec.select, i32 9
  %trunc = trunc i32 %tmp1.2 to i6
  br label %bb9

bb9:                                              ; preds = %bb5, %bb19
  %tmp2.03 = phi i32 [ 0, %bb5 ], [ %tmp21, %bb19 ]
  switch i6 %trunc, label %bb24 [
    i6 0, label %bb19
    i6 -15, label %bb22
  ]

bb19:                                             ; preds = %bb9
  %tmp21 = add nuw nsw i32 %tmp2.03, 1
  %tmp8 = icmp eq i32 %tmp21, 25
  br i1 %tmp8, label %bb22, label %bb9

bb22:                                             ; preds = %bb19, %bb9
  unreachable

bb24:                                             ; preds = %bb9
  %tmp29 = or i32 %tmp3.0, 1
  br label %bb5
}
