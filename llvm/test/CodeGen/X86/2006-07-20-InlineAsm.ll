; RUN: llc < %s -mtriple=i686-- -no-integrated-as
; PR833

@G = weak global i32 0		; <ptr> [#uses=3]

define i32 @foo(i32 %X) {
entry:
	%X_addr = alloca i32		; <ptr> [#uses=3]
	store i32 %X, ptr %X_addr
	call void asm sideeffect "xchg{l} {$0,$1|$1,$0}", "=*m,=*r,m,1,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @G, ptr elementtype(i32) %X_addr, ptr @G, i32 %X )
	%tmp1 = load i32, ptr %X_addr		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32 @foo2(i32 %X) {
entry:
	%X_addr = alloca i32		; <ptr> [#uses=3]
	store i32 %X, ptr %X_addr
	call void asm sideeffect "xchg{l} {$0,$1|$1,$0}", "=*m,=*r,1,~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @G, ptr elementtype(i32) %X_addr, i32 %X )
	%tmp1 = load i32, ptr %X_addr		; <i32> [#uses=1]
	ret i32 %tmp1
}

