; RUN: opt < %s -instcombine -S | grep "ret i1 false"
; PR2697

define i1 @x(i32 %x) nounwind {
	%div = sdiv i32 %x, 65536		; <i32> [#uses=1]
	%cmp = icmp slt i32 %div, -65536
	ret i1 %cmp
}
