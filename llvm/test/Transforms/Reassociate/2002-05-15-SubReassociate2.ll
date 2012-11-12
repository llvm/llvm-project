; With sub reassociation, constant folding can eliminate the two 12 constants.
;
; RUN: opt < %s -reassociate -constprop -dce -S | not grep 12

define i32 @test(i32 %A, i32 %B, i32 %C, i32 %D) {
	%M = add i32 %A, 12		; <i32> [#uses=1]
	%N = add i32 %M, %B		; <i32> [#uses=1]
	%O = add i32 %N, %C		; <i32> [#uses=1]
	%P = sub i32 %D, %O		; <i32> [#uses=1]
	%Q = add i32 %P, 12		; <i32> [#uses=1]
	ret i32 %Q
}

