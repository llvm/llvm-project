; RUN: llvm-as < %s | opt -reassociate -dce | llvm-dis | \
; RUN:   not grep {\\(and\\|sub\\)}

define i32 @test1(i32 %a, i32 %b) {
	%tmp.2 = and i32 %b, %a		; <i32> [#uses=1]
	%tmp.4 = xor i32 %a, -1		; <i32> [#uses=1]
        ; (A&B)&~A == 0
	%tmp.5 = and i32 %tmp.2, %tmp.4		; <i32> [#uses=1]
	ret i32 %tmp.5
}

define i32 @test2(i32 %a, i32 %b) {
	%tmp.1 = and i32 %a, 1234		; <i32> [#uses=1]
	%tmp.2 = and i32 %b, %tmp.1		; <i32> [#uses=1]
	%tmp.4 = xor i32 %a, -1		; <i32> [#uses=1]
	; A&~A == 0
        %tmp.5 = and i32 %tmp.2, %tmp.4		; <i32> [#uses=1]
	ret i32 %tmp.5
}

define i32 @test3(i32 %b, i32 %a) {
	%tmp.1 = add i32 %a, 1234		; <i32> [#uses=1]
	%tmp.2 = add i32 %b, %tmp.1		; <i32> [#uses=1]
	%tmp.4 = sub i32 0, %a		; <i32> [#uses=1]
        ; (b+(a+1234))+-a -> b+1234
  	%tmp.5 = add i32 %tmp.2, %tmp.4		; <i32> [#uses=1]
	ret i32 %tmp.5
}
