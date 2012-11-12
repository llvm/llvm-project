; RUN: opt < %s -inline -S | grep call
; Do not inline calls to variable-sized alloca.

@q = common global i8* null		; <i8**> [#uses=1]

define i8* @a(i32 %i) nounwind {
entry:
	%i_addr = alloca i32		; <i32*> [#uses=2]
	%retval = alloca i8*		; <i8**> [#uses=1]
	%p = alloca i8*		; <i8**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %i, i32* %i_addr
	%0 = load i32* %i_addr, align 4		; <i32> [#uses=1]
	%1 = alloca i8, i32 %0		; <i8*> [#uses=1]
	store i8* %1, i8** %p, align 4
	%2 = load i8** %p, align 4		; <i8*> [#uses=1]
	store i8* %2, i8** @q, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i8** %retval		; <i8*> [#uses=1]
	ret i8* %retval1
}

define void @b(i32 %i) nounwind {
entry:
	%i_addr = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i32 %i, i32* %i_addr
	%0 = load i32* %i_addr, align 4		; <i32> [#uses=1]
	%1 = call i8* @a(i32 %0) nounwind		; <i8*> [#uses=0]
	br label %return

return:		; preds = %entry
	ret void
}
