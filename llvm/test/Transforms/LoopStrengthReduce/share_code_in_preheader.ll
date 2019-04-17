; RUN: opt < %s -loop-reduce -S | grep mul | count 1
; LSR should not make two copies of the Q*L expression in the preheader!

define i8 @test(i8* %A, i8* %B, i32 %L, i32 %Q, i32 %N.s) {
entry:
	%tmp.6 = mul i32 %Q, %L		; <i32> [#uses=1]
	%N = bitcast i32 %N.s to i32		; <i32> [#uses=1]
	br label %no_exit
no_exit:		; preds = %no_exit, %entry
	%indvar.ui = phi i32 [ 0, %entry ], [ %indvar.next, %no_exit ]		; <i32> [#uses=2]
	%Sum.0.0 = phi i8 [ 0, %entry ], [ %tmp.21, %no_exit ]		; <i8> [#uses=1]
	%indvar = bitcast i32 %indvar.ui to i32		; <i32> [#uses=1]
	%N_addr.0.0 = sub i32 %N.s, %indvar		; <i32> [#uses=1]
	%tmp.8 = add i32 %N_addr.0.0, %tmp.6		; <i32> [#uses=2]
	%tmp.9 = getelementptr i8, i8* %A, i32 %tmp.8		; <i8*> [#uses=1]
	%tmp.10 = load i8, i8* %tmp.9		; <i8> [#uses=1]
	%tmp.17 = getelementptr i8, i8* %B, i32 %tmp.8		; <i8*> [#uses=1]
	%tmp.18 = load i8, i8* %tmp.17		; <i8> [#uses=1]
	%tmp.19 = sub i8 %tmp.10, %tmp.18		; <i8> [#uses=1]
	%tmp.21 = add i8 %tmp.19, %Sum.0.0		; <i8> [#uses=2]
	%indvar.next = add i32 %indvar.ui, 1		; <i32> [#uses=2]
	%exitcond = icmp eq i32 %indvar.next, %N		; <i1> [#uses=1]
	br i1 %exitcond, label %loopexit, label %no_exit
loopexit:		; preds = %no_exit
	ret i8 %tmp.21
}

