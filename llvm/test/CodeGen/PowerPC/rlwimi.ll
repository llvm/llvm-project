; All of these ands and shifts should be folded into rlwimi's
; RUN: llc < %s -march=ppc32 | not grep and
; RUN: llc < %s -march=ppc32 | grep rlwimi | count 8

define i32 @test1(i32 %x, i32 %y) {
entry:
	%tmp.3 = shl i32 %x, 16		; <i32> [#uses=1]
	%tmp.7 = and i32 %y, 65535		; <i32> [#uses=1]
	%tmp.9 = or i32 %tmp.7, %tmp.3		; <i32> [#uses=1]
	ret i32 %tmp.9
}

define i32 @test2(i32 %x, i32 %y) {
entry:
	%tmp.7 = and i32 %x, 65535		; <i32> [#uses=1]
	%tmp.3 = shl i32 %y, 16		; <i32> [#uses=1]
	%tmp.9 = or i32 %tmp.7, %tmp.3		; <i32> [#uses=1]
	ret i32 %tmp.9
}

define i32 @test3(i32 %x, i32 %y) {
entry:
	%tmp.3 = lshr i32 %x, 16		; <i32> [#uses=1]
	%tmp.6 = and i32 %y, -65536		; <i32> [#uses=1]
	%tmp.7 = or i32 %tmp.6, %tmp.3		; <i32> [#uses=1]
	ret i32 %tmp.7
}

define i32 @test4(i32 %x, i32 %y) {
entry:
	%tmp.6 = and i32 %x, -65536		; <i32> [#uses=1]
	%tmp.3 = lshr i32 %y, 16		; <i32> [#uses=1]
	%tmp.7 = or i32 %tmp.6, %tmp.3		; <i32> [#uses=1]
	ret i32 %tmp.7
}

define i32 @test5(i32 %x, i32 %y) {
entry:
	%tmp.3 = shl i32 %x, 1		; <i32> [#uses=1]
	%tmp.4 = and i32 %tmp.3, -65536		; <i32> [#uses=1]
	%tmp.7 = and i32 %y, 65535		; <i32> [#uses=1]
	%tmp.9 = or i32 %tmp.4, %tmp.7		; <i32> [#uses=1]
	ret i32 %tmp.9
}

define i32 @test6(i32 %x, i32 %y) {
entry:
	%tmp.7 = and i32 %x, 65535		; <i32> [#uses=1]
	%tmp.3 = shl i32 %y, 1		; <i32> [#uses=1]
	%tmp.4 = and i32 %tmp.3, -65536		; <i32> [#uses=1]
	%tmp.9 = or i32 %tmp.4, %tmp.7		; <i32> [#uses=1]
	ret i32 %tmp.9
}

define i32 @test7(i32 %x, i32 %y) {
entry:
	%tmp.2 = and i32 %x, -65536		; <i32> [#uses=1]
	%tmp.5 = and i32 %y, 65535		; <i32> [#uses=1]
	%tmp.7 = or i32 %tmp.5, %tmp.2		; <i32> [#uses=1]
	ret i32 %tmp.7
}

define i32 @test8(i32 %bar) {
entry:
	%tmp.3 = shl i32 %bar, 1		; <i32> [#uses=1]
	%tmp.4 = and i32 %tmp.3, 2		; <i32> [#uses=1]
	%tmp.6 = and i32 %bar, -3		; <i32> [#uses=1]
	%tmp.7 = or i32 %tmp.4, %tmp.6		; <i32> [#uses=1]
	ret i32 %tmp.7
}
