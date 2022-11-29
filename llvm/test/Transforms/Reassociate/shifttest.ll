; With shl->mul reassociation, we can see that this is (shl A, 9) * A
;
; RUN: opt < %s -passes=reassociate,instcombine -S |\
; RUN:    grep "shl .*, 9"

define i32 @test(i32 %A, i32 %B) {
	%X = shl i32 %A, 5		; <i32> [#uses=1]
	%Y = shl i32 %A, 4		; <i32> [#uses=1]
	%Z = mul i32 %Y, %X		; <i32> [#uses=1]
	ret i32 %Z
}

