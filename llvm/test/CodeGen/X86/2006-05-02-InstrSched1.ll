; REQUIRES: asserts
; RUN: llc < %s -mtriple=i686-unknown-linux -relocation-model=static -stats 2>&1 | \
; RUN:   grep asm-printer | grep 14
;
; It's possible to schedule this in 14 instructions by avoiding
; callee-save registers, but the scheduler isn't currently that
; conervative with registers.
@size20 = external dso_local global i32		; <ptr> [#uses=1]
@in5 = external dso_local global ptr		; <ptr> [#uses=1]

define i32 @compare(ptr %a, ptr %b) nounwind {
	%tmp.upgrd.1 = load i32, ptr @size20		; <i32> [#uses=1]
	%tmp.upgrd.2 = load ptr, ptr @in5		; <ptr> [#uses=2]
	%tmp3 = load i32, ptr %b		; <i32> [#uses=1]
	%gep.upgrd.3 = zext i32 %tmp3 to i64		; <i64> [#uses=1]
	%tmp4 = getelementptr i8, ptr %tmp.upgrd.2, i64 %gep.upgrd.3		; <ptr> [#uses=2]
	%tmp7 = load i32, ptr %a		; <i32> [#uses=1]
	%gep.upgrd.4 = zext i32 %tmp7 to i64		; <i64> [#uses=1]
	%tmp8 = getelementptr i8, ptr %tmp.upgrd.2, i64 %gep.upgrd.4		; <ptr> [#uses=2]
	%tmp.upgrd.5 = tail call i32 @memcmp( ptr %tmp8, ptr %tmp4, i32 %tmp.upgrd.1 )		; <i32> [#uses=1]
	ret i32 %tmp.upgrd.5
}

declare i32 @memcmp(ptr, ptr, i32)
