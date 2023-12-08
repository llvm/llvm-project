; This test makes sure that the xor instructions are properly eliminated
; when arbitrary precision integers are used.

; RUN: opt < %s -passes=instcombine -S | not grep xor

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

