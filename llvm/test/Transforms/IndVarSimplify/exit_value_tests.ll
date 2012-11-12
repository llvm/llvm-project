; Test that we can evaluate the exit values of various expression types.  Since
; these loops all have predictable exit values we can replace the use outside
; of the loop with a closed-form computation, making the loop dead.
;
; RUN: opt < %s -indvars -loop-deletion -simplifycfg | \
; RUN:   llvm-dis | not grep br

define i32 @polynomial_constant() {
; <label>:0
	br label %Loop

Loop:		; preds = %Loop, %0
	%A1 = phi i32 [ 0, %0 ], [ %A2, %Loop ]		; <i32> [#uses=3]
	%B1 = phi i32 [ 0, %0 ], [ %B2, %Loop ]		; <i32> [#uses=1]
	%A2 = add i32 %A1, 1		; <i32> [#uses=1]
	%B2 = add i32 %B1, %A1		; <i32> [#uses=2]
	%C = icmp eq i32 %A1, 1000		; <i1> [#uses=1]
	br i1 %C, label %Out, label %Loop

Out:		; preds = %Loop
	ret i32 %B2
}

define i32 @NSquare(i32 %N) {
; <label>:0
	br label %Loop

Loop:		; preds = %Loop, %0
	%X = phi i32 [ 0, %0 ], [ %X2, %Loop ]		; <i32> [#uses=4]
	%X2 = add i32 %X, 1		; <i32> [#uses=1]
	%c = icmp eq i32 %X, %N		; <i1> [#uses=1]
	br i1 %c, label %Out, label %Loop

Out:		; preds = %Loop
	%Y = mul i32 %X, %X		; <i32> [#uses=1]
	ret i32 %Y
}

define i32 @NSquareOver2(i32 %N) {
; <label>:0
	br label %Loop

Loop:		; preds = %Loop, %0
	%X = phi i32 [ 0, %0 ], [ %X2, %Loop ]		; <i32> [#uses=3]
	%Y = phi i32 [ 15, %0 ], [ %Y2, %Loop ]		; <i32> [#uses=1]
	%Y2 = add i32 %Y, %X		; <i32> [#uses=2]
	%X2 = add i32 %X, 1		; <i32> [#uses=1]
	%c = icmp eq i32 %X, %N		; <i1> [#uses=1]
	br i1 %c, label %Out, label %Loop

Out:		; preds = %Loop
	ret i32 %Y2
}

define i32 @strength_reduced() {
; <label>:0
	br label %Loop

Loop:		; preds = %Loop, %0
	%A1 = phi i32 [ 0, %0 ], [ %A2, %Loop ]		; <i32> [#uses=3]
	%B1 = phi i32 [ 0, %0 ], [ %B2, %Loop ]		; <i32> [#uses=1]
	%A2 = add i32 %A1, 1		; <i32> [#uses=1]
	%B2 = add i32 %B1, %A1		; <i32> [#uses=2]
	%C = icmp eq i32 %A1, 1000		; <i1> [#uses=1]
	br i1 %C, label %Out, label %Loop

Out:		; preds = %Loop
	ret i32 %B2
}

define i32 @chrec_equals() {
entry:
	br label %no_exit

no_exit:		; preds = %no_exit, %entry
	%i0 = phi i32 [ 0, %entry ], [ %i1, %no_exit ]		; <i32> [#uses=3]
	%ISq = mul i32 %i0, %i0		; <i32> [#uses=1]
	%i1 = add i32 %i0, 1		; <i32> [#uses=2]
	%tmp.1 = icmp ne i32 %ISq, 10000		; <i1> [#uses=1]
	br i1 %tmp.1, label %no_exit, label %loopexit

loopexit:		; preds = %no_exit
	ret i32 %i1
}

define i16 @cast_chrec_test() {
; <label>:0
	br label %Loop

Loop:		; preds = %Loop, %0
	%A1 = phi i32 [ 0, %0 ], [ %A2, %Loop ]		; <i32> [#uses=2]
	%B1 = trunc i32 %A1 to i16		; <i16> [#uses=2]
	%A2 = add i32 %A1, 1		; <i32> [#uses=1]
	%C = icmp eq i16 %B1, 1000		; <i1> [#uses=1]
	br i1 %C, label %Out, label %Loop

Out:		; preds = %Loop
	ret i16 %B1
}

define i32 @linear_div_fold() {
entry:
	br label %loop

loop:		; preds = %loop, %entry
	%i = phi i32 [ 4, %entry ], [ %i.next, %loop ]		; <i32> [#uses=3]
	%i.next = add i32 %i, 8		; <i32> [#uses=1]
	%RV = udiv i32 %i, 2		; <i32> [#uses=1]
	%c = icmp ne i32 %i, 68		; <i1> [#uses=1]
	br i1 %c, label %loop, label %loopexit

loopexit:		; preds = %loop
	ret i32 %RV
}
