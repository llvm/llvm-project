; This input caused the mem2reg pass to die because it was trying to promote
; the %r alloca, even though it is invalid to do so in this case!
;
; RUN: llvm-as < %s | opt -mem2reg

define void @test() {
	%r = alloca i32		; <i32*> [#uses=2]
	store i32 4, i32* %r
	store i32* %r, i32** null
	ret void
}

