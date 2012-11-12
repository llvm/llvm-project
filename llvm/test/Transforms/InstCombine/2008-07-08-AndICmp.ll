; RUN: opt < %s -instcombine -S | grep icmp | count 1
; PR2330

define i1 @foo(i32 %a, i32 %b) nounwind {
entry:
	icmp ult i32 %a, 8		; <i1>:0 [#uses=1]
	icmp ult i32 %b, 8		; <i1>:1 [#uses=1]
	and i1 %1, %0		; <i1>:2 [#uses=1]
	ret i1 %2
}
