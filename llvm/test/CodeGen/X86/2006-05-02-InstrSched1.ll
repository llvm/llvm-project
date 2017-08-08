; REQUIRES: asserts
; RUN: llc < %s -mtriple=i686-unknown-linux -relocation-model=static -stats 2>&1 | \
; RUN:   grep asm-printer | grep 14
;
; It's possible to schedule this in 14 instructions by avoiding
; callee-save registers, but the scheduler isn't currently that
; conervative with registers.
@size20 = external global i32		; <i32*> [#uses=1]
@in5 = external global i8*		; <i8**> [#uses=1]

define i32 @compare(i8* %a, i8* %b) nounwind {
	%tmp = bitcast i8* %a to i32*		; <i32*> [#uses=1]
	%tmp1 = bitcast i8* %b to i32*		; <i32*> [#uses=1]
	%tmp.upgrd.1 = load i32, i32* @size20		; <i32> [#uses=1]
	%tmp.upgrd.2 = load i8*, i8** @in5		; <i8*> [#uses=2]
	%tmp3 = load i32, i32* %tmp1		; <i32> [#uses=1]
	%gep.upgrd.3 = zext i32 %tmp3 to i64		; <i64> [#uses=1]
	%tmp4 = getelementptr i8, i8* %tmp.upgrd.2, i64 %gep.upgrd.3		; <i8*> [#uses=2]
	%tmp7 = load i32, i32* %tmp		; <i32> [#uses=1]
	%gep.upgrd.4 = zext i32 %tmp7 to i64		; <i64> [#uses=1]
	%tmp8 = getelementptr i8, i8* %tmp.upgrd.2, i64 %gep.upgrd.4		; <i8*> [#uses=2]
	%tmp.upgrd.5 = tail call i32 @memcmp( i8* %tmp8, i8* %tmp4, i32 %tmp.upgrd.1 )		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.5
}

declare i32 @memcmp(i8*, i8*, i32)
