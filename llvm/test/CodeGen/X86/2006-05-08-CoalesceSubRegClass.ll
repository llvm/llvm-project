; Coalescing from R32 to a subset R32_. Once another register coalescer bug is
; fixed, the movb should go away as well.

; RUN: llvm-as < %s | llc -march=x86 -relocation-model=static | \
; RUN:   grep movl

@B = external global i32		; <i32*> [#uses=2]
@C = external global i16*		; <i16**> [#uses=2]

define void @test(i32 %A) {
	%A.upgrd.1 = trunc i32 %A to i8		; <i8> [#uses=1]
	%tmp2 = load i32* @B		; <i32> [#uses=1]
	%tmp3 = and i8 %A.upgrd.1, 16		; <i8> [#uses=1]
	%shift.upgrd.2 = zext i8 %tmp3 to i32		; <i32> [#uses=1]
	%tmp4 = shl i32 %tmp2, %shift.upgrd.2		; <i32> [#uses=1]
	store i32 %tmp4, i32* @B
	%tmp6 = lshr i32 %A, 3		; <i32> [#uses=1]
	%tmp = load i16** @C		; <i16*> [#uses=1]
	%tmp8 = ptrtoint i16* %tmp to i32		; <i32> [#uses=1]
	%tmp9 = add i32 %tmp8, %tmp6		; <i32> [#uses=1]
	%tmp9.upgrd.3 = inttoptr i32 %tmp9 to i16*		; <i16*> [#uses=1]
	store i16* %tmp9.upgrd.3, i16** @C
	ret void
}

