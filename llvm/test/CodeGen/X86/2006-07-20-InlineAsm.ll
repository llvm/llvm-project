; RUN: llvm-as < %s | llc -march=x86
; PR833

@G = weak global i32 0		; <i32*> [#uses=3]

define i32 @foo(i32 %X) {
entry:
	%X_addr = alloca i32		; <i32*> [#uses=3]
	store i32 %X, i32* %X_addr
	call void asm sideeffect "xchg{l} {$0,$1|$1,$0}", "=*m,=*r,m,1,~{dirflag},~{fpsr},~{flags}"( i32* @G, i32* %X_addr, i32* @G, i32 %X )
	%tmp1 = load i32* %X_addr		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32 @foo2(i32 %X) {
entry:
	%X_addr = alloca i32		; <i32*> [#uses=3]
	store i32 %X, i32* %X_addr
	call void asm sideeffect "xchg{l} {$0,$1|$1,$0}", "=*m,=*r,1,~{dirflag},~{fpsr},~{flags}"( i32* @G, i32* %X_addr, i32 %X )
	%tmp1 = load i32* %X_addr		; <i32> [#uses=1]
	ret i32 %tmp1
}

