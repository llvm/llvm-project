; XFAIL: *
; REQUIRES: asserts
; RUN: opt < %s -passes='unswitch<nontrivial>' -disable-output
; RUN: opt < %s -simple-loop-unswitch -enable-nontrivial-unswitch -disable-output


; Make sure we don't crash due to a dangling use of %tmp2 in bb7.
define void @test.use_in_dead_block(i1 %arg1, i1 %arg2) {
bb1:
  br label %bb2

bb2:                                              ; preds = %bb4, %bb1
  %tmp1 = phi i64 [ 0, %bb4 ], [ 42, %bb1 ]
  br i1 %arg1, label %bb3, label %bb6

bb3:                                              ; preds = %bb2
  br i1 %arg2, label %bb5, label %bb4

bb4:                                              ; preds = %bb3
  %tmp2 = add i32 1, 1
  br label %bb2

bb5:                                             ; preds = %bb3
  ret void

bb6:                                             ; preds = %bb2
  %tmp3 = add i64 %tmp1, 1
  ret void

bb7:                                             ; No predecessors!
  %tmp4 = add i32 %tmp2, 1
  ret void
}
