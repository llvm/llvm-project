; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {icmp ne i32 \%a}
; PR2330

define i1 @foo(i32 %a) nounwind  {
entry:
	%tmp15 = shl i32 1, %a		; <i32> [#uses=1]
	%tmp237 = and i32 %tmp15, 1		; <i32> [#uses=1]
	%toBool = icmp eq i32 %tmp237, 0		; <i1> [#uses=1]
	ret i1 %toBool
}
