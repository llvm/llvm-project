; RUN: llc < %s -march=ppc32 -mtriple=powerpc-apple-darwin8 | \
; RUN:    grep cntlzw

define i32 @foo() nounwind {
entry:
	%retval = alloca i32, align 4		; <i32*> [#uses=2]
	%temp = alloca i32, align 4		; <i32*> [#uses=2]
	%ctz_x = alloca i32, align 4		; <i32*> [#uses=3]
	%ctz_c = alloca i32, align 4		; <i32*> [#uses=2]
	store i32 61440, i32* %ctz_x
	%tmp = load i32* %ctz_x		; <i32> [#uses=1]
	%tmp1 = sub i32 0, %tmp		; <i32> [#uses=1]
	%tmp2 = load i32* %ctz_x		; <i32> [#uses=1]
	%tmp3 = and i32 %tmp1, %tmp2		; <i32> [#uses=1]
	%tmp4 = call i32 asm "$(cntlz$|cntlzw$) $0,$1", "=r,r,~{dirflag},~{fpsr},~{flags}"( i32 %tmp3 )		; <i32> [#uses=1]
	store i32 %tmp4, i32* %ctz_c
	%tmp5 = load i32* %ctz_c		; <i32> [#uses=1]
	store i32 %tmp5, i32* %temp
	%tmp6 = load i32* %temp		; <i32> [#uses=1]
	store i32 %tmp6, i32* %retval
	br label %return

return:		; preds = %entry
	%retval2 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval2
}
