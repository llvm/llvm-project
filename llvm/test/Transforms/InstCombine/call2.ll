; RUN: opt < %s -passes=instcombine | llvm-dis

; This used to crash trying to do a double-to-pointer conversion
define i32 @bar() {
entry:
	%retval = alloca i32, align 4		; <ptr> [#uses=1]
	%tmp = call i32 (...) @f( double 3.000000e+00 )		; <i32> [#uses=0]
	br label %return

return:		; preds = %entry
	%retval1 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval1
}

define i32 @f(ptr %p) {
entry:
	%p_addr = alloca ptr		; <ptr> [#uses=1]
	%retval = alloca i32, align 4		; <ptr> [#uses=1]
	store ptr %p, ptr %p_addr
	br label %return

return:		; preds = %entry
	%retval1 = load i32, ptr %retval		; <i32> [#uses=1]
	ret i32 %retval1
}
