; RUN: llc < %s -march=x86

; ModuleID = 'a.bc'

define i32 @foo(i32 %A, i32 %B) {
entry:
	%A_addr = alloca i32		; <i32*> [#uses=2]
	%B_addr = alloca i32		; <i32*> [#uses=1]
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%tmp = alloca i32, align 4		; <i32*> [#uses=2]
	%ret = alloca i32, align 4		; <i32*> [#uses=2]
	store i32 %A, i32* %A_addr
	store i32 %B, i32* %B_addr
	%tmp1 = load i32* %A_addr		; <i32> [#uses=1]
	%tmp2 = call i32 asm "roll $1,$0", "=r,I,0,~{dirflag},~{fpsr},~{flags},~{cc}"( i32 7, i32 %tmp1 )		; <i32> [#uses=1]
	store i32 %tmp2, i32* %ret
	%tmp3 = load i32* %ret		; <i32> [#uses=1]
	store i32 %tmp3, i32* %tmp
	%tmp4 = load i32* %tmp		; <i32> [#uses=1]
	store i32 %tmp4, i32* %retval
	br label %return

return:		; preds = %entry
	%retval5 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval5
}
