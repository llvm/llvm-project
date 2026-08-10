; RUN: llc < %s 
; PR2267
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @atomic_store_rel_int(ptr %p, i32 %v) nounwind  {
entry:
	%asmtmp = tail call i32 asm sideeffect "xchgl $1,$0", "=*m,=r,*m,1,~{dirflag},~{fpsr},~{flags}"( ptr elementtype( i32) %p, ptr elementtype(i32) %p, i32 %v ) nounwind 		; <i32> [#uses=0]
	ret void
}

