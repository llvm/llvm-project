; Reassociation should apply to Add, Mul, And, Or, & Xor
;
; RUN: opt < %s -reassociate -constprop -instcombine -die -S | not grep 12

define i32 @test_mul(i32 %arg) {
	%tmp1 = mul i32 12, %arg		; <i32> [#uses=1]
	%tmp2 = mul i32 %tmp1, 12		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test_and(i32 %arg) {
	%tmp1 = and i32 14, %arg		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 14		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test_or(i32 %arg) {
	%tmp1 = or i32 14, %arg		; <i32> [#uses=1]
	%tmp2 = or i32 %tmp1, 14		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test_xor(i32 %arg) {
	%tmp1 = xor i32 12, %arg		; <i32> [#uses=1]
	%tmp2 = xor i32 %tmp1, 12		; <i32> [#uses=1]
	ret i32 %tmp2
}

