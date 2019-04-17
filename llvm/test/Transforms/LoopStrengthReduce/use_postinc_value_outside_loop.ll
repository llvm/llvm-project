; RUN: opt < %s -loop-reduce -S | \
; RUN:   grep "add i32 %indvar630.ui, 1"
;
; Make sure that the use of the IV outside of the loop (the store) uses the 
; post incremented value of the IV, not the preincremented value.  This 
; prevents the loop from having to keep the post and pre-incremented value
; around for the duration of the loop, adding a copy and an extra register
; to the loop.

declare i1 @pred(i32)

define void @test([700 x i32]* %nbeaux_.0__558, i32* %i_.16574) {
then.0:
	br label %no_exit.2
no_exit.2:		; preds = %no_exit.2, %then.0
	%indvar630.ui = phi i32 [ 0, %then.0 ], [ %indvar.next631, %no_exit.2 ]		; <i32> [#uses=3]
	%indvar630 = bitcast i32 %indvar630.ui to i32		; <i32> [#uses=2]
	%gep.upgrd.1 = zext i32 %indvar630.ui to i64		; <i64> [#uses=1]
	%tmp.38 = getelementptr [700 x i32], [700 x i32]* %nbeaux_.0__558, i32 0, i64 %gep.upgrd.1		; <i32*> [#uses=1]
	store i32 0, i32* %tmp.38
	%inc.2 = add i32 %indvar630, 2		; <i32> [#uses=1]
	%tmp.34 = call i1 @pred( i32 %indvar630 )		; <i1> [#uses=1]
	%indvar.next631 = add i32 %indvar630.ui, 1		; <i32> [#uses=1]
	br i1 %tmp.34, label %no_exit.2, label %loopexit.2.loopexit
loopexit.2.loopexit:		; preds = %no_exit.2
	store i32 %inc.2, i32* %i_.16574
	ret void
}

