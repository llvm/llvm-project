; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    egrep {shl|lshr|ashr} | count 3

define i41 @test0(i41 %A, i41 %B, i41 %C) {
	%X = shl i41 %A, %C
	%Y = shl i41 %B, %C
	%Z = and i41 %X, %Y
	ret i41 %Z
}

define i57 @test1(i57 %A, i57 %B, i57 %C) {
	%X = lshr i57 %A, %C
	%Y = lshr i57 %B, %C
	%Z = or i57 %X, %Y
	ret i57 %Z
}

define i49 @test2(i49 %A, i49 %B, i49 %C) {
	%X = ashr i49 %A, %C
	%Y = ashr i49 %B, %C
	%Z = xor i49 %X, %Y
	ret i49 %Z
}
