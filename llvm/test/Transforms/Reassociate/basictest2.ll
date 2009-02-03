; With reassociation, constant folding can eliminate the +/- 30 constants.
;
; RUN: llvm-as < %s | opt -reassociate -constprop -instcombine -die | llvm-dis | not grep 30

define i32 @test(i32 %reg109, i32 %reg1111) {
	%reg115 = add i32 %reg109, -30		; <i32> [#uses=1]
	%reg116 = add i32 %reg115, %reg1111		; <i32> [#uses=1]
	%reg117 = add i32 %reg116, 30		; <i32> [#uses=1]
	ret i32 %reg117
}

