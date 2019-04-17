; RUN: opt < %s -loop-reduce -S | \
; RUN:   not grep "bitcast i32 1 to i32"
; END.
; The setlt wants to use a value that is incremented one more than the dominant
; IV.  Don't insert the 1 outside the loop, preventing folding it into the add.

define void @test([700 x i32]* %nbeaux_.0__558, i32* %i_.16574) {
then.0:
	br label %no_exit.2
no_exit.2:		; preds = %no_exit.2, %then.0
	%indvar630 = phi i32 [ 0, %then.0 ], [ %indvar.next631, %no_exit.2 ]		; <i32> [#uses=4]
	%gep.upgrd.1 = zext i32 %indvar630 to i64		; <i64> [#uses=1]
	%tmp.38 = getelementptr [700 x i32], [700 x i32]* %nbeaux_.0__558, i32 0, i64 %gep.upgrd.1		; <i32*> [#uses=1]
	store i32 0, i32* %tmp.38
	%inc.2 = add i32 %indvar630, 2		; <i32> [#uses=2]
	%tmp.34 = icmp slt i32 %inc.2, 701		; <i1> [#uses=1]
	%indvar.next631 = add i32 %indvar630, 1		; <i32> [#uses=1]
	br i1 %tmp.34, label %no_exit.2, label %loopexit.2.loopexit
loopexit.2.loopexit:		; preds = %no_exit.2
	store i32 %inc.2, i32* %i_.16574
	ret void
}

