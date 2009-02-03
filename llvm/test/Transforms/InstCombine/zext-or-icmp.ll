; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep icmp | count 1

	%struct.FooBar = type <{ i8, i8, [2 x i8], i8, i8, i8, i8, i16, i16, [4 x i8], [8 x %struct.Rock] }>
	%struct.Rock = type { i16, i16 }
@some_idx = internal constant [4 x i8] c"\0A\0B\0E\0F"		; <[4 x i8]*> [#uses=1]

define i8 @t(%struct.FooBar* %up, i8 zeroext  %intra_flag, i32 %blk_i) zeroext nounwind  {
entry:
	%tmp2 = lshr i32 %blk_i, 1		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp2, 2		; <i32> [#uses=1]
	%tmp5 = and i32 %blk_i, 1		; <i32> [#uses=1]
	%tmp6 = or i32 %tmp3, %tmp5		; <i32> [#uses=1]
	%tmp8 = getelementptr %struct.FooBar* %up, i32 0, i32 7		; <i16*> [#uses=1]
	%tmp9 = load i16* %tmp8, align 1		; <i16> [#uses=1]
	%tmp910 = zext i16 %tmp9 to i32		; <i32> [#uses=1]
	%tmp12 = getelementptr [4 x i8]* @some_idx, i32 0, i32 %tmp6		; <i8*> [#uses=1]
	%tmp13 = load i8* %tmp12, align 1		; <i8> [#uses=1]
	%tmp1314 = zext i8 %tmp13 to i32		; <i32> [#uses=1]
	%tmp151 = lshr i32 %tmp910, %tmp1314		; <i32> [#uses=1]
	%tmp1516 = trunc i32 %tmp151 to i8		; <i8> [#uses=1]
	%tmp18 = getelementptr %struct.FooBar* %up, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp19 = load i8* %tmp18, align 1		; <i8> [#uses=1]
	%tmp22 = and i8 %tmp1516, %tmp19		; <i8> [#uses=1]
	%tmp24 = getelementptr %struct.FooBar* %up, i32 0, i32 0		; <i8*> [#uses=1]
	%tmp25 = load i8* %tmp24, align 1		; <i8> [#uses=1]
	%tmp26.mask = and i8 %tmp25, 1		; <i8> [#uses=1]
	%toBool = icmp eq i8 %tmp26.mask, 0		; <i1> [#uses=1]
	%toBool.not = xor i1 %toBool, true		; <i1> [#uses=1]
	%toBool33 = icmp eq i8 %intra_flag, 0		; <i1> [#uses=1]
	%bothcond = or i1 %toBool.not, %toBool33		; <i1> [#uses=1]
	%iftmp.1.0 = select i1 %bothcond, i8 0, i8 1		; <i8> [#uses=1]
	%tmp40 = or i8 %tmp22, %iftmp.1.0		; <i8> [#uses=1]
	%tmp432 = and i8 %tmp40, 1		; <i8> [#uses=1]
	ret i8 %tmp432
}
