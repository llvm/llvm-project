; RUN: opt < %s -licm | llvm-dis
; RUN: opt -aa-pipeline=basic-aa -passes='require<aa>,require<targetir>,require<scalar-evolution>,require<opt-remark-emit>,loop-mssa(licm)' < %s | llvm-dis

define void @testfunc(i32 %i) {
; <label>:0
	br label %Loop
Loop:		; preds = %Loop, %0
	%j = phi i32 [ 0, %0 ], [ %Next, %Loop ]		; <i32> [#uses=1]
	%i2 = mul i32 %i, 17		; <i32> [#uses=1]
	%Next = add i32 %j, %i2		; <i32> [#uses=2]
	%cond = icmp eq i32 %Next, 0		; <i1> [#uses=1]
	br i1 %cond, label %Out, label %Loop
Out:		; preds = %Loop
	ret void
}

