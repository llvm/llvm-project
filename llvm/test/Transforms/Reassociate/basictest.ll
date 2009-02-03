; With reassociation, constant folding can eliminate the 12 and -12 constants.
;
; RUN: llvm-as < %s | opt -reassociate -constprop -instcombine -die | llvm-dis | not grep add

define i32 @test(i32 %arg) {
	%tmp1 = sub i32 -12, %arg		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 12		; <i32> [#uses=1]
	ret i32 %tmp2
}

