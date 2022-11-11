; PR672
; RUN: %lli -jit-kind=mcjit %s
; RUN: %lli %s
; XFAIL: i686 && windows

define i32 @main() {
	%f = bitcast i32 (i32, i32*, i32)* @check_tail to i32*		; <i32*> [#uses=1]
	%res = tail call fastcc i32 @check_tail( i32 10, i32* %f, i32 10 )		; <i32> [#uses=1]
	ret i32 %res
}

define fastcc i32 @check_tail(i32 %x, i32* %f, i32 %g) {
	%tmp1 = icmp sgt i32 %x, 0		; <i1> [#uses=1]
	br i1 %tmp1, label %if-then, label %if-else
if-then:		; preds = %0
	%fun_ptr = bitcast i32* %f to i32 (i32, i32*, i32)*		; <i32 (i32, i32*, i32)*> [#uses=1]
	%arg1 = add i32 %x, -1		; <i32> [#uses=1]
	%res = tail call fastcc i32 %fun_ptr( i32 %arg1, i32* %f, i32 %g )		; <i32> [#uses=1]
	ret i32 %res
if-else:		; preds = %0
	ret i32 %x
}

