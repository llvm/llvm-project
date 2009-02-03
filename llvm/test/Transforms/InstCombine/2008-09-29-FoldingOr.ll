; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {or i1}
; PR2844

define i32 @test(i32 %p_74) {
	%A = icmp eq i32 %p_74, 0		; <i1> [#uses=1]
	%B = icmp slt i32 %p_74, -638208501		; <i1> [#uses=1]
	%or.cond = or i1 %A, %B		; <i1> [#uses=1]
	%iftmp.10.0 = select i1 %or.cond, i32 0, i32 1		; <i32> [#uses=1]
	ret i32 %iftmp.10.0
}
