; There should be exactly one shift and one add left.
; RUN: opt < %s -reassociate -instcombine -S > %t
; RUN: grep shl %t | count 1
; RUN: grep add %t | count 1

define i32 @test(i32 %X, i32 %Y) {
	%tmp.2 = shl i32 %X, 1		; <i32> [#uses=1]
	%tmp.6 = shl i32 %Y, 1		; <i32> [#uses=1]
	%tmp.4 = add i32 %tmp.6, %tmp.2		; <i32> [#uses=1]
	ret i32 %tmp.4
}

