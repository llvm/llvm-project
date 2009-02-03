; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {and i32 %a, 1} | count 4
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {or i32 %0, %b} | count 4


define i32 @func1(i32 %a, i32 %b) nounwind readnone {
entry:
	%0 = or i32 %b, %a		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%2 = and i32 %b, -2		; <i32> [#uses=1]
	%3 = or i32 %1, %2		; <i32> [#uses=1]
	ret i32 %3
}

define i32 @func2(i32 %a, i32 %b) nounwind readnone {
entry:
	%0 = or i32 %a, %b		; <i32> [#uses=1]
	%1 = and i32 1, %0		; <i32> [#uses=1]
	%2 = and i32 -2, %b		; <i32> [#uses=1]
	%3 = or i32 %1, %2		; <i32> [#uses=1]
	ret i32 %3
}

define i32 @func3(i32 %a, i32 %b) nounwind readnone {
entry:
	%0 = or i32 %b, %a		; <i32> [#uses=1]
	%1 = and i32 %0, 1		; <i32> [#uses=1]
	%2 = and i32 %b, -2		; <i32> [#uses=1]
	%3 = or i32 %2, %1		; <i32> [#uses=1]
	ret i32 %3
}

define i32 @func4(i32 %a, i32 %b) nounwind readnone {
entry:
	%0 = or i32 %a, %b		; <i32> [#uses=1]
	%1 = and i32 1, %0		; <i32> [#uses=1]
	%2 = and i32 -2, %b		; <i32> [#uses=1]
	%3 = or i32 %2, %1		; <i32> [#uses=1]
	ret i32 %3
}
