; RUN: llvm-as < %s | llc -march=x86 -regalloc=local | grep {subl	%eax, %edx}

; Local regalloc shouldn't assume that both the uses of the
; sub instruction are kills, because one of them is tied
; to an output. Previously, it was allocating both inputs
; in the same register.

define i32 @func_3() nounwind {
entry:
	%retval = alloca i32		; <i32*> [#uses=2]
	%g_323 = alloca i8		; <i8*> [#uses=2]
	%p_5 = alloca i64, align 8		; <i64*> [#uses=2]
	%0 = alloca i32		; <i32*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store i64 0, i64* %p_5, align 8
	store i8 1, i8* %g_323, align 1
	%1 = load i8* %g_323, align 1		; <i8> [#uses=1]
	%2 = sext i8 %1 to i64		; <i64> [#uses=1]
	%3 = load i64* %p_5, align 8		; <i64> [#uses=1]
	%4 = sub i64 %3, %2		; <i64> [#uses=1]
	%5 = icmp sge i64 %4, 0		; <i1> [#uses=1]
	%6 = zext i1 %5 to i32		; <i32> [#uses=1]
	store i32 %6, i32* %0, align 4
	%7 = load i32* %0, align 4		; <i32> [#uses=1]
	store i32 %7, i32* %retval, align 4
	br label %return

return:		; preds = %entry
	%retval1 = load i32* %retval		; <i32> [#uses=1]
	ret i32 %retval1
}
