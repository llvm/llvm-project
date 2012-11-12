; RUN: opt < %s -reassociate -instcombine -S | not grep sub

; Test that we can turn things like X*-(Y*Z) -> X*-1*Y*Z.

define i32 @test1(i32 %a, i32 %b, i32 %z) {
	%c = sub i32 0, %z		; <i32> [#uses=1]
	%d = mul i32 %a, %b		; <i32> [#uses=1]
	%e = mul i32 %c, %d		; <i32> [#uses=1]
	%f = mul i32 %e, 12345		; <i32> [#uses=1]
	%g = sub i32 0, %f		; <i32> [#uses=1]
	ret i32 %g
}

define i32 @test2(i32 %a, i32 %b, i32 %z) {
	%d = mul i32 %z, 40		; <i32> [#uses=1]
	%c = sub i32 0, %d		; <i32> [#uses=1]
	%e = mul i32 %a, %c		; <i32> [#uses=1]
	%f = sub i32 0, %e		; <i32> [#uses=1]
	ret i32 %f
}

