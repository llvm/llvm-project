; This test makes sure that the xor instructions are properly eliminated
; when arbitrary precision integers are used.

; RUN: opt < %s -instcombine -S | not grep xor

define i33 @test1(i33 %A) {
	%B = xor i33 %A, -1
	%C = xor i33 %B, -1
	ret i33 %C
}

define i1 @test2(i52 %A, i52 %B) {
	%cond = icmp ule i52 %A, %B     ; Can change into uge
	%Ret = xor i1 %cond, true
	ret i1 %Ret
}

; Test that demorgans law can be instcombined
define i47 @test3(i47 %A, i47 %B) {
	%a = xor i47 %A, -1
	%b = xor i47 %B, -1
	%c = and i47 %a, %b
	%d = xor i47 %c, -1
	ret i47 %d
}

; Test that demorgens law can work with constants
define i61 @test4(i61 %A, i61 %B) {
	%a = xor i61 %A, -1
	%c = and i61 %a, 5    ; 5 = ~c2
	%d = xor i61 %c, -1
	ret i61 %d
}

; test the mirror of demorgans law...
define i71 @test5(i71 %A, i71 %B) {
	%a = xor i71 %A, -1
	%b = xor i71 %B, -1
	%c = or i71 %a, %b
	%d = xor i71 %c, -1
	ret i71 %d
}
