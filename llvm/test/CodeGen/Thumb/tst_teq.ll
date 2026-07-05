; RUN: llc -mtriple=thumb-eabi %s -o - | FileCheck %s

define i32 @f(i32 %a) {
entry:
	%tmp2 = and i32 %a, 255		; <i32> [#uses=1]
	icmp eq i32 %tmp2, 0		; <i1>:0 [#uses=1]
	%retval = select i1 %0, i32 20, i32 10		; <i32> [#uses=1]
	ret i32 %retval
}

define i32 @g(i32 %a) {
entry:
        %tmp2 = xor i32 %a, 255
	icmp eq i32 %tmp2, 0		; <i1>:0 [#uses=1]
	%retval = select i1 %0, i32 20, i32 10		; <i32> [#uses=1]
	ret i32 %retval
}

; CHECK: tst

