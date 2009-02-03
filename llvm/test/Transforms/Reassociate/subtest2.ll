; With sub reassociation, constant folding can eliminate the uses of %a.
;
; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis | grep %a | count 1
; PR2047

define i32 @test(i32 %a, i32 %b, i32 %c) nounwind  {
entry:
	%tmp3 = sub i32 %a, %b		; <i32> [#uses=1]
	%tmp5 = sub i32 %tmp3, %c		; <i32> [#uses=1]
	%tmp7 = sub i32 %tmp5, %a		; <i32> [#uses=1]
	ret i32 %tmp7
}

