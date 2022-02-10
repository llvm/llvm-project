; RUN: opt < %s -passes=instcombine | llvm-dis

; This used to crash trying to do a double-to-pointer conversion
define i32 @bar() {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=1]
	%tmp = call i32 (...) bitcast (i32 (i8*)* @f to i32 (...)*)( double 3.000000e+00 )		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	%retval1 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval1
}

define i32 @f(i8* %p) {
entry:
	%p_addr = alloca i8*		; <i8**> [#uses=1]
	%retval = alloca i32, align 4		; <i32*> [#uses=1]
	store i8* %p, i8** %p_addr
	br label %return

return:		; preds = %entry
	%retval1 = load i32, i32* %retval		; <i32> [#uses=1]
	ret i32 %retval1
}
