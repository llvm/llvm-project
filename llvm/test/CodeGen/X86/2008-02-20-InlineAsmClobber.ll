; RUN: llc < %s | grep "a:" | not grep ax
; RUN: llc < %s | grep "b:" | not grep ax
; PR2078
; The clobber list says that "ax" is clobbered.  Make sure that eax isn't 
; allocated to the input/output register.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@pixels = weak global i32 0		; <i32*> [#uses=2]

define void @test() nounwind  {
entry:
	%tmp = load i32* @pixels, align 4		; <i32> [#uses=1]
	%tmp1 = tail call i32 asm sideeffect "a: $0 $1", "=r,0,~{dirflag},~{fpsr},~{flags},~{ax}"( i32 %tmp ) nounwind 		; <i32> [#uses=1]
	store i32 %tmp1, i32* @pixels, align 4
	ret void
}

define void @test2(i16* %block, i8* %pixels, i32 %line_size) nounwind  {
entry:
	%tmp1 = getelementptr i16* %block, i32 64		; <i16*> [#uses=1]
	%tmp3 = tail call i8* asm sideeffect "b: $0 $1 $2", "=r,r,0,~{dirflag},~{fpsr},~{flags},~{ax}"( i16* %tmp1, i8* %pixels ) nounwind 		; <i8*> [#uses=0]
	ret void
}

