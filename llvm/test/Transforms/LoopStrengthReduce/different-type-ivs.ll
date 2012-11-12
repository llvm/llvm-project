; RUN: opt < %s -loop-reduce -disable-output
; Test to make sure that loop-reduce never crashes on IV's 
; with different types but identical strides.

define void @foo() {
entry:
	br label %no_exit
no_exit:		; preds = %no_exit, %entry
	%indvar = phi i32 [ 0, %entry ], [ %indvar.next, %no_exit ]		; <i32> [#uses=3]
	%indvar.upgrd.1 = trunc i32 %indvar to i16		; <i16> [#uses=1]
	%X.0.0 = mul i16 %indvar.upgrd.1, 1234		; <i16> [#uses=1]
	%tmp. = mul i32 %indvar, 1234		; <i32> [#uses=1]
	%tmp.5 = sext i16 %X.0.0 to i32		; <i32> [#uses=1]
	%tmp.3 = call i32 (...)* @bar( i32 %tmp.5, i32 %tmp. )		; <i32> [#uses=0]
	%tmp.0 = call i1 @pred( )		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %tmp.0, label %return, label %no_exit
return:		; preds = %no_exit
	ret void
}

declare i1 @pred()

declare i32 @bar(...)

