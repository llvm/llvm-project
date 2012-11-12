; RUN: llc < %s -mtriple=armv6-apple-darwin | \
; RUN:   grep uxt | count 10

define i32 @test1(i32 %x) {
	%tmp1 = and i32 %x, 16711935		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32 @test2(i32 %x) {
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test3(i32 %x) {
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test4(i32 %x) {
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp6 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test5(i32 %x) {
	%tmp1 = lshr i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711935		; <i32> [#uses=1]
	ret i32 %tmp2
}

define i32 @test6(i32 %x) {
	%tmp1 = lshr i32 %x, 16		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 255		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 16		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test7(i32 %x) {
	%tmp1 = lshr i32 %x, 16		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 255		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 16		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test8(i32 %x) {
	%tmp1 = shl i32 %x, 8		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16711680		; <i32> [#uses=1]
	%tmp5 = lshr i32 %x, 24		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test9(i32 %x) {
	%tmp1 = lshr i32 %x, 24		; <i32> [#uses=1]
	%tmp4 = shl i32 %x, 8		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp5, %tmp1		; <i32> [#uses=1]
	ret i32 %tmp6
}

define i32 @test10(i32 %p0) {
	%tmp1 = lshr i32 %p0, 7		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 16253176		; <i32> [#uses=2]
	%tmp4 = lshr i32 %tmp2, 5		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 458759		; <i32> [#uses=1]
	%tmp7 = or i32 %tmp5, %tmp2		; <i32> [#uses=1]
	ret i32 %tmp7
}
