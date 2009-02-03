; RUN: llvm-as < %s | opt -reassociate -instcombine | llvm-dis |\
; RUN:   not grep {sub i32 0}

define i32 @test(i32 %X, i32 %Y, i32 %Z) {
	%A = sub i32 0, %X		; <i32> [#uses=1]
	%B = mul i32 %A, %Y		; <i32> [#uses=1]
        ; (-X)*Y + Z -> Z-X*Y
	%C = add i32 %B, %Z		; <i32> [#uses=1]
	ret i32 %C
}
