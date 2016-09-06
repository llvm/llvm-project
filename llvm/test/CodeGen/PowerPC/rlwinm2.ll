; All of these ands and shifts should be folded into rlw[i]nm instructions
; RUN: llc -verify-machineinstrs < %s -march=ppc32 -o %t
; RUN: not grep and %t
; RUN: not grep srawi %t 
; RUN: not grep srwi %t 
; RUN: not grep slwi %t 
; RUN: grep rlwnm %t | count 1
; RUN: grep rlwinm %t | count 1

define i32 @test1(i32 %X, i32 %Y) {
entry:
	%tmp = trunc i32 %Y to i8		; <i8> [#uses=2]
	%tmp1 = shl i32 %X, %Y		; <i32> [#uses=1]
	%tmp2 = sub i32 32, %Y		; <i8> [#uses=1]
	%tmp3 = lshr i32 %X, %tmp2		; <i32> [#uses=1]
	%tmp4 = or i32 %tmp1, %tmp3		; <i32> [#uses=1]
	%tmp6 = and i32 %tmp4, 127		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test2(i32 %X) {
entry:
	%tmp1 = lshr i32 %X, 27		; <i32> [#uses=1]
	%tmp2 = shl i32 %X, 5		; <i32> [#uses=1]
	%tmp2.masked = and i32 %tmp2, 96		; <i32> [#uses=1]
	%tmp5 = or i32 %tmp1, %tmp2.masked		; <i32> [#uses=1]
	ret i32 %tmp5
}
