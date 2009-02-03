; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | grep mul | count 2

; This should have exactly 2 multiplies when we're done.

define i32 @f(i32 %a, i32 %b) {
	%tmp.2 = mul i32 %a, %a		; <i32> [#uses=1]
	%tmp.5 = shl i32 %a, 1		; <i32> [#uses=1]
	%tmp.6 = mul i32 %tmp.5, %b		; <i32> [#uses=1]
	%tmp.10 = mul i32 %b, %b		; <i32> [#uses=1]
	%tmp.7 = add i32 %tmp.6, %tmp.2		; <i32> [#uses=1]
	%tmp.11 = add i32 %tmp.7, %tmp.10		; <i32> [#uses=1]
	ret i32 %tmp.11
}

