; RUN: llvm-as < %s | opt -licm | llvm-dis | %prcontext volatile 1 | grep Loop

@X = global i32 7		; <i32*> [#uses=2]

define void @testfunc(i32 %i) {
	br label %Loop
Loop:		; preds = %Loop, %0
        ; Should not promote this to a register
	%x = volatile load i32* @X		; <i32> [#uses=1]
	%x2 = add i32 %x, 1		; <i32> [#uses=1]
	store i32 %x2, i32* @X
	br i1 true, label %Out, label %Loop
Out:		; preds = %Loop
	ret void
}

