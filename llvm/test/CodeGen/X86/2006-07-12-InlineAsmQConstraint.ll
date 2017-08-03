; RUN: llc < %s -march=x86
; PR828

target datalayout = "e-p:32:32"
target triple = "i686-pc-linux-gnu"

define void @_ZN5() {
cond_true9:
	%tmp3.i.i = call i32 asm sideeffect "lock; cmpxchg $1,$2", "={ax},q,m,0,~{dirflag},~{fpsr},~{flags},~{memory}"( i32 0, i32* null, i32 0 )		; <i32> [#uses=0]
	ret void
}

