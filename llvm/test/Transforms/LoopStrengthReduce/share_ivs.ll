; RUN: opt < %s -loop-reduce -S | grep phi | count 1

; This testcase should have ONE stride 18 indvar, the other use should have a
; loop invariant value (B) added to it inside of the loop, instead of having
; a whole indvar based on B for it.

declare i1 @cond(i32)

define void @test(i32 %B) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%IV = phi i32 [ 0, %0 ], [ %IVn, %Loop ]		; <i32> [#uses=3]
	%C = mul i32 %IV, 18		; <i32> [#uses=1]
	%D = mul i32 %IV, 18		; <i32> [#uses=1]
	%E = add i32 %D, %B		; <i32> [#uses=1]
	%cnd = call i1 @cond( i32 %E )		; <i1> [#uses=1]
	call i1 @cond( i32 %C )		; <i1>:1 [#uses=0]
	%IVn = add i32 %IV, 1		; <i32> [#uses=1]
	br i1 %cnd, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}

