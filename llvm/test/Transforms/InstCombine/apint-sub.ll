; This test makes sure that sub instructions are properly eliminated
; even with arbitrary precision integers.
;

; RUN: opt < %s -instcombine -S | \
; RUN:   grep -v "sub i19 %Cok, %Bok" | grep -v "sub i25 0, %Aok" | not grep sub
; END.

define i23 @test1(i23 %A) {
	%B = sub i23 %A, %A		; <i23> [#uses=1]
	ret i23 %B
}

define i47 @test2(i47 %A) {
	%B = sub i47 %A, 0		; <i47> [#uses=1]
	ret i47 %B
}

define i97 @test3(i97 %A) {
	%B = sub i97 0, %A		; <i97> [#uses=1]
	%C = sub i97 0, %B		; <i97> [#uses=1]
	ret i97 %C
}

define i108 @test4(i108 %A, i108 %x) {
	%B = sub i108 0, %A		; <i108> [#uses=1]
	%C = sub i108 %x, %B		; <i108> [#uses=1]
	ret i108 %C
}

define i19 @test5(i19 %A, i19 %Bok, i19 %Cok) {
	%D = sub i19 %Bok, %Cok		; <i19> [#uses=1]
	%E = sub i19 %A, %D		; <i19> [#uses=1]
	ret i19 %E
}

define i57 @test6(i57 %A, i57 %B) {
	%C = and i57 %A, %B		; <i57> [#uses=1]
	%D = sub i57 %A, %C		; <i57> [#uses=1]
	ret i57 %D
}

define i77 @test7(i77 %A) {
	%B = sub i77 -1, %A		; <i77> [#uses=1]
	ret i77 %B
}

define i27 @test8(i27 %A) {
	%B = mul i27 9, %A		; <i27> [#uses=1]
	%C = sub i27 %B, %A		; <i27> [#uses=1]
	ret i27 %C
}

define i42 @test9(i42 %A) {
	%B = mul i42 3, %A		; <i42> [#uses=1]
	%C = sub i42 %A, %B		; <i42> [#uses=1]
	ret i42 %C
}

define i124 @test10(i124 %A, i124 %B) {
	%C = sub i124 0, %A		; <i124> [#uses=1]
	%D = sub i124 0, %B		; <i124> [#uses=1]
	%E = mul i124 %C, %D		; <i124> [#uses=1]
	ret i124 %E
}

define i55 @test10a(i55 %A) {
	%C = sub i55 0, %A		; <i55> [#uses=1]
	%E = mul i55 %C, 7		; <i55> [#uses=1]
	ret i55 %E
}

define i1 @test11(i9 %A, i9 %B) {
	%C = sub i9 %A, %B		; <i9> [#uses=1]
	%cD = icmp ne i9 %C, 0		; <i1> [#uses=1]
	ret i1 %cD
}

define i43 @test12(i43 %A) {
	%B = ashr i43 %A, 42		; <i43> [#uses=1]
	%C = sub i43 0, %B		; <i43> [#uses=1]
	ret i43 %C
}

define i79 @test13(i79 %A) {
	%B = lshr i79 %A, 78		; <i79> [#uses=1]
	%C = sub i79 0, %B		; <i79> [#uses=1]
	ret i79 %C
}

define i1024 @test14(i1024 %A) {
	%B = lshr i1024 %A, 1023        ; <i1024> [#uses=1]
	%C = bitcast i1024 %B to i1024	; <i1024> [#uses=1]
	%D = sub i1024 0, %C		; <i1024> [#uses=1]
	ret i1024 %D
}

define i14 @test15(i14 %A, i14 %B) {
	%C = sub i14 0, %A		; <i14> [#uses=1]
	%D = srem i14 %B, %C		; <i14> [#uses=1]
	ret i14 %D
}

define i51 @test16(i51 %A) {
	%X = sdiv i51 %A, 1123		; <i51> [#uses=1]
	%Y = sub i51 0, %X		; <i51> [#uses=1]
	ret i51 %Y
}

; Can't fold subtract here because negation it might oveflow.
; PR3142
define i25 @test17(i25 %Aok) {
	%B = sub i25 0, %Aok		; <i25> [#uses=1]
	%C = sdiv i25 %B, 1234		; <i25> [#uses=1]
	ret i25 %C
}

define i128 @test18(i128 %Y) {
	%tmp.4 = shl i128 %Y, 2		; <i128> [#uses=1]
	%tmp.12 = shl i128 %Y, 2	; <i128> [#uses=1]
	%tmp.8 = sub i128 %tmp.4, %tmp.12	; <i128> [#uses=1]
	ret i128 %tmp.8
}

define i39 @test19(i39 %X, i39 %Y) {
	%Z = sub i39 %X, %Y		; <i39> [#uses=1]
	%Q = add i39 %Z, %Y		; <i39> [#uses=1]
	ret i39 %Q
}

define i1 @test20(i33 %g, i33 %h) {
	%tmp.2 = sub i33 %g, %h		; <i33> [#uses=1]
	%tmp.4 = icmp ne i33 %tmp.2, %g		; <i1> [#uses=1]
	ret i1 %tmp.4
}

define i1 @test21(i256 %g, i256 %h) {
	%tmp.2 = sub i256 %g, %h	; <i256> [#uses=1]
	%tmp.4 = icmp ne i256 %tmp.2, %g; <i1> [#uses=1]
	ret i1 %tmp.4
}
