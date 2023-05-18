; RUN: opt -passes="verify<scalar-evolution>,lcssa,verify<scalar-evolution>" -verify-scev-strict -S -disable-output %s

; The first SCEV verification is required because it queries SCEV and populates
; SCEV caches. Second SCEV verification checks if the caches are in valid state.

; Check that the second SCEV verification doesn't fail.
define void @test(ptr %arg, ptr %arg1, i1 %arg2, i1 %arg3) {
bb:
  br label %bb6

bb5:
  br label %bb6

bb6:
  br label %bb7

bb7:
  %tmp = load i32, ptr %arg
  %tmp8 = load i32, ptr %arg1
  %tmp9 = add i32 %tmp8, %tmp
  %tmp10 = icmp sgt i32 %tmp9, %tmp
  br i1 %tmp10, label %bb11, label %bb17

bb11:
  br i1 %arg3, label %bb12, label %bb14

bb12:
  br label %bb13

bb13:
  br label %bb17

bb14:
  br label %bb15

bb15:
  %tmp16 = add nsw i32 %tmp, 1
  ret void

bb17:
  %tmp18 = phi i32 [ 0, %bb7 ], [ %tmp8, %bb13 ]
  br i1 %arg2, label %bb24, label %bb19

bb19:
  br label %bb20

bb20:
  %tmp21 = phi i32 [ %tmp22, %bb20 ], [ 0, %bb19 ]
  %tmp22 = add nuw nsw i32 %tmp21, 1
  %tmp23 = icmp slt i32 %tmp22, %tmp18
  br i1 %tmp23, label %bb20, label %bb5

bb24:
  ret void
}
