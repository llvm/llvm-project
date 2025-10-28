; RUN: opt < %s -passes='require<domfrontier>,loop-simplify,require<domfrontier>' -verify-dom-info


define void @a(i1 %arg) nounwind {
entry:
  br i1 %arg, label %bb37, label %bb1.i

bb1.i:                                            ; preds = %bb1.i, %bb
  %indvar = phi i64 [ %indvar.next, %bb1.i ], [ 0, %entry ] ; <i64> [#uses=1]
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, 576       ; <i1> [#uses=1]
  br i1 %exitcond, label %bb37, label %bb1.i

bb37:                                             ; preds = %bb1.i, %bb
  br label %return


return:                                           ; preds = %bb39
  ret void
}
