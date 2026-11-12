; RUN: opt -passes="verify<scalar-evolution>,lcssa,verify<scalar-evolution>" -verify-scev-strict -S %s

; The first SCEV verification is required because it queries SCEV and populates
; SCEV caches. Second SCEV verification checks if the caches are in valid state.

; Check that the second SCEV verification doesn't fail.
define void @foo(ptr %arg, ptr %arg1, i1 %arg2) {
bb:
  br label %bb3

bb3:                                              ; preds = %bb13, %bb
  %tmp = load i32, ptr %arg
  %tmp4 = load i32, ptr %arg1
  %tmp5 = add i32 %tmp4, %tmp
  %tmp6 = icmp sgt i32 %tmp5, %tmp
  br i1 %tmp6, label %bb7, label %bb11

bb7:                                              ; preds = %bb3
  br i1 %arg2, label %bb10, label %bb8

bb8:                                              ; preds = %bb7
  %tmp9 = add nsw i32 %tmp, 1
  ret void

bb10:                                             ; preds = %bb7
  br label %bb11

bb11:                                             ; preds = %bb10, %bb3
  %tmp12 = phi i32 [ 0, %bb3 ], [ %tmp4, %bb10 ]
  br label %bb13

bb13:                                             ; preds = %bb13, %bb11
  %tmp14 = phi i32 [ %tmp15, %bb13 ], [ 0, %bb11 ]
  %tmp15 = add nuw nsw i32 %tmp14, 1
  %tmp16 = icmp slt i32 %tmp15, %tmp12
  br i1 %tmp16, label %bb13, label %bb3
}
