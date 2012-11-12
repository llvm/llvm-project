; All of these ands and shifts should be folded into rlwimi's
; RUN: llc < %s -march=ppc32 -o %t
; RUN: not grep and %t
; RUN: not grep srawi %t
; RUN: not grep srwi %t
; RUN: not grep slwi %t
; RUN: grep rlwinm %t | count 8

define i32 @test1(i32 %a) {
entry:
	%tmp.1 = and i32 %a, 268431360		; <i32> [#uses=1]
	ret i32 %tmp.1
}

define i32 @test2(i32 %a) {
entry:
	%tmp.1 = and i32 %a, -268435441		; <i32> [#uses=1]
	ret i32 %tmp.1
}

define i32 @test3(i32 %a) {
entry:
	%tmp.2 = ashr i32 %a, 8		; <i32> [#uses=1]
	%tmp.3 = and i32 %tmp.2, 255		; <i32> [#uses=1]
	ret i32 %tmp.3
}

define i32 @test4(i32 %a) {
entry:
	%tmp.3 = lshr i32 %a, 8		; <i32> [#uses=1]
	%tmp.4 = and i32 %tmp.3, 255		; <i32> [#uses=1]
	ret i32 %tmp.4
}

define i32 @test5(i32 %a) {
entry:
	%tmp.2 = shl i32 %a, 8		; <i32> [#uses=1]
	%tmp.3 = and i32 %tmp.2, -8388608		; <i32> [#uses=1]
	ret i32 %tmp.3
}

define i32 @test6(i32 %a) {
entry:
	%tmp.1 = and i32 %a, 65280		; <i32> [#uses=1]
	%tmp.2 = ashr i32 %tmp.1, 8		; <i32> [#uses=1]
	ret i32 %tmp.2
}

define i32 @test7(i32 %a) {
entry:
	%tmp.1 = and i32 %a, 65280		; <i32> [#uses=1]
	%tmp.2 = lshr i32 %tmp.1, 8		; <i32> [#uses=1]
	ret i32 %tmp.2
}

define i32 @test8(i32 %a) {
entry:
	%tmp.1 = and i32 %a, 16711680		; <i32> [#uses=1]
	%tmp.2 = shl i32 %tmp.1, 8		; <i32> [#uses=1]
	ret i32 %tmp.2
}
