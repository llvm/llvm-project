; Don't reduce the byte access to P[i], at least not on targets that 
; support an efficient 'mem[r1+r2]' addressing mode.

; RUN: opt < %s -loop-reduce -disable-output


declare i1 @pred(i32)

define void @test(ptr %PTR) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%INDVAR = phi i32 [ 0, %0 ], [ %INDVAR2, %Loop ]		; <i32> [#uses=2]
	%STRRED = getelementptr i8, ptr %PTR, i32 %INDVAR		; <ptr> [#uses=1]
	store i8 0, ptr %STRRED
	%INDVAR2 = add i32 %INDVAR, 1		; <i32> [#uses=2]
        ;; cannot eliminate indvar
	%cond = call i1 @pred( i32 %INDVAR2 )		; <i1> [#uses=1]
	br i1 %cond, label %Loop, label %Out
Out:		; preds = %Loop
	ret void
}
