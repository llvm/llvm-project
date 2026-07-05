; RUN: opt < %s -passes=loop-rotate -verify-dom-info -verify-loop-info -verify-memoryssa -disable-output

define void @func(i1 %arg) {
bb0:
  br label %bb1

bb1:                                              ; preds = %bb4, %bb0
  %0 = phi i16 [ %2, %bb4 ], [ 0, %bb0 ]
  %1 = icmp sle i16 %0, 2
  br i1 %1, label %bb2, label %bb5

bb2:                                              ; preds = %bb1
  br i1 %arg, label %bb6, label %bb4

bb3:                                              ; No predecessors!
  br label %bb6

bb4:                                              ; preds = %bb2
  %2 = add i16 undef, 1
  br label %bb1

bb5:                                              ; preds = %bb1
  br label %bb6

bb6:                                              ; preds = %bb5, %bb3, %bb2
  unreachable
}
