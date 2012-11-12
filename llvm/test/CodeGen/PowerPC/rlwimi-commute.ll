; RUN: llc < %s -march=ppc32 | grep rlwimi
; RUN: llc < %s -march=ppc32 | not grep "or "

; Make sure there is no register-register copies here.

define void @test1(i32* %A, i32* %B, i32* %D, i32* %E) {
	%A.upgrd.1 = load i32* %A		; <i32> [#uses=2]
	%B.upgrd.2 = load i32* %B		; <i32> [#uses=1]
	%X = and i32 %A.upgrd.1, 15		; <i32> [#uses=1]
	%Y = and i32 %B.upgrd.2, -16		; <i32> [#uses=1]
	%Z = or i32 %X, %Y		; <i32> [#uses=1]
	store i32 %Z, i32* %D
	store i32 %A.upgrd.1, i32* %E
	ret void
}

define void @test2(i32* %A, i32* %B, i32* %D, i32* %E) {
	%A.upgrd.3 = load i32* %A		; <i32> [#uses=1]
	%B.upgrd.4 = load i32* %B		; <i32> [#uses=2]
	%X = and i32 %A.upgrd.3, 15		; <i32> [#uses=1]
	%Y = and i32 %B.upgrd.4, -16		; <i32> [#uses=1]
	%Z = or i32 %X, %Y		; <i32> [#uses=1]
	store i32 %Z, i32* %D
	store i32 %B.upgrd.4, i32* %E
	ret void
}

define i32 @test3(i32 %a, i32 %b) {
	%tmp.1 = and i32 %a, 15		; <i32> [#uses=1]
	%tmp.3 = and i32 %b, 240		; <i32> [#uses=1]
	%tmp.4 = or i32 %tmp.3, %tmp.1		; <i32> [#uses=1]
	ret i32 %tmp.4
}

