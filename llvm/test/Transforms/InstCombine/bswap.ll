; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    grep {call.*llvm.bswap} | count 6

define i32 @test1(i32 %i) {
	%tmp1 = lshr i32 %i, 24		; <i32> [#uses=1]
	%tmp3 = lshr i32 %i, 8		; <i32> [#uses=1]
	%tmp4 = and i32 %tmp3, 65280		; <i32> [#uses=1]
	%tmp5 = or i32 %tmp1, %tmp4		; <i32> [#uses=1]
	%tmp7 = shl i32 %i, 8		; <i32> [#uses=1]
	%tmp8 = and i32 %tmp7, 16711680		; <i32> [#uses=1]
	%tmp9 = or i32 %tmp5, %tmp8		; <i32> [#uses=1]
	%tmp11 = shl i32 %i, 24		; <i32> [#uses=1]
	%tmp12 = or i32 %tmp9, %tmp11		; <i32> [#uses=1]
	ret i32 %tmp12
}

define i32 @test2(i32 %arg) {
	%tmp2 = shl i32 %arg, 24		; <i32> [#uses=1]
	%tmp4 = shl i32 %arg, 8		; <i32> [#uses=1]
	%tmp5 = and i32 %tmp4, 16711680		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp2, %tmp5		; <i32> [#uses=1]
	%tmp8 = lshr i32 %arg, 8		; <i32> [#uses=1]
	%tmp9 = and i32 %tmp8, 65280		; <i32> [#uses=1]
	%tmp10 = or i32 %tmp6, %tmp9		; <i32> [#uses=1]
	%tmp12 = lshr i32 %arg, 24		; <i32> [#uses=1]
	%tmp14 = or i32 %tmp10, %tmp12		; <i32> [#uses=1]
	ret i32 %tmp14
}

define i16 @test3(i16 %s) {
	%tmp2 = lshr i16 %s, 8		; <i16> [#uses=1]
	%tmp4 = shl i16 %s, 8		; <i16> [#uses=1]
	%tmp5 = or i16 %tmp2, %tmp4		; <i16> [#uses=1]
	ret i16 %tmp5
}

define i16 @test4(i16 %s) {
	%tmp2 = lshr i16 %s, 8		; <i16> [#uses=1]
	%tmp4 = shl i16 %s, 8		; <i16> [#uses=1]
	%tmp5 = or i16 %tmp4, %tmp2		; <i16> [#uses=1]
	ret i16 %tmp5
}

define i16 @test5(i16 %a) {
	%tmp = zext i16 %a to i32		; <i32> [#uses=2]
	%tmp1 = and i32 %tmp, 65280		; <i32> [#uses=1]
	%tmp2 = ashr i32 %tmp1, 8		; <i32> [#uses=1]
	%tmp2.upgrd.1 = trunc i32 %tmp2 to i16		; <i16> [#uses=1]
	%tmp4 = and i32 %tmp, 255		; <i32> [#uses=1]
	%tmp5 = shl i32 %tmp4, 8		; <i32> [#uses=1]
	%tmp5.upgrd.2 = trunc i32 %tmp5 to i16		; <i16> [#uses=1]
	%tmp.upgrd.3 = or i16 %tmp2.upgrd.1, %tmp5.upgrd.2		; <i16> [#uses=1]
	%tmp6 = bitcast i16 %tmp.upgrd.3 to i16		; <i16> [#uses=1]
	%tmp6.upgrd.4 = zext i16 %tmp6 to i32		; <i32> [#uses=1]
	%retval = trunc i32 %tmp6.upgrd.4 to i16		; <i16> [#uses=1]
	ret i16 %retval
}

; PR2842
define i32 @test6(i32 %x) nounwind readnone {
	%tmp = shl i32 %x, 16		; <i32> [#uses=1]
	%x.mask = and i32 %x, 65280		; <i32> [#uses=1]
	%tmp1 = lshr i32 %x, 16		; <i32> [#uses=1]
	%tmp2 = and i32 %tmp1, 255		; <i32> [#uses=1]
	%tmp3 = or i32 %x.mask, %tmp		; <i32> [#uses=1]
	%tmp4 = or i32 %tmp3, %tmp2		; <i32> [#uses=1]
	%tmp5 = shl i32 %tmp4, 8		; <i32> [#uses=1]
	%tmp6 = lshr i32 %x, 24		; <i32> [#uses=1]
	%tmp7 = or i32 %tmp5, %tmp6		; <i32> [#uses=1]
	ret i32 %tmp7
}

