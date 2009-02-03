; RUN: llvm-as < %s | opt  -licm -disable-output -stats |& \
; RUN:    grep {memory locations promoted to register}
@X = global i32 7		; <i32*> [#uses=4]

define void @testfunc(i32 %i) {
; <label>:0
	br label %Loop

Loop:		; preds = %Loop, %0
	%j = phi i32 [ 0, %0 ], [ %Next, %Loop ]		; <i32> [#uses=1]
	%x = load i32* @X		; <i32> [#uses=1]
	%x2 = add i32 %x, 1		; <i32> [#uses=1]
	store i32 %x2, i32* @X
	%Next = add i32 %j, 1		; <i32> [#uses=2]
	%cond = icmp eq i32 %Next, 0		; <i1> [#uses=1]
	br i1 %cond, label %Out, label %Loop

Out:		; preds = %Loop
	ret void
}

define void @testhard(i32 %i) {
	br label %Loop

Loop:		; preds = %Loop, %0
	%X1 = getelementptr i32* @X, i64 0		; <i32*> [#uses=1]
	%A = load i32* %X1		; <i32> [#uses=1]
	%V = add i32 %A, 1		; <i32> [#uses=1]
	%X2 = getelementptr i32* @X, i64 0		; <i32*> [#uses=1]
	store i32 %V, i32* %X2
	br i1 false, label %Loop, label %Exit

Exit:		; preds = %Loop
	ret void
}
