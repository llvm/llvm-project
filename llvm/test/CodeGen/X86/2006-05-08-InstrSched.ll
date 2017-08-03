; RUN: llc < %s -march=x86 -relocation-model=static | not grep "subl.*%esp"

@A = external global i16*		; <i16**> [#uses=1]
@B = external global i32		; <i32*> [#uses=1]
@C = external global i32		; <i32*> [#uses=2]

define void @test() {
	%tmp = load i16*, i16** @A		; <i16*> [#uses=1]
	%tmp1 = getelementptr i16, i16* %tmp, i32 1		; <i16*> [#uses=1]
	%tmp.upgrd.1 = load i16, i16* %tmp1		; <i16> [#uses=1]
	%tmp3 = zext i16 %tmp.upgrd.1 to i32		; <i32> [#uses=1]
	%tmp.upgrd.2 = load i32, i32* @B		; <i32> [#uses=1]
	%tmp4 = and i32 %tmp.upgrd.2, 16		; <i32> [#uses=1]
	%tmp5 = load i32, i32* @C		; <i32> [#uses=1]
	%tmp6 = trunc i32 %tmp4 to i8		; <i8> [#uses=2]
	%shift.upgrd.3 = zext i8 %tmp6 to i32		; <i32> [#uses=1]
	%tmp7 = shl i32 %tmp5, %shift.upgrd.3		; <i32> [#uses=1]
	%tmp9 = xor i8 %tmp6, 16		; <i8> [#uses=1]
	%shift.upgrd.4 = zext i8 %tmp9 to i32		; <i32> [#uses=1]
	%tmp11 = lshr i32 %tmp3, %shift.upgrd.4		; <i32> [#uses=1]
	%tmp12 = or i32 %tmp11, %tmp7		; <i32> [#uses=1]
	store i32 %tmp12, i32* @C
	ret void
}

