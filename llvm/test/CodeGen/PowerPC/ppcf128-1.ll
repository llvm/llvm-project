; RUN: opt < %s -O3 | llc > %t
; ModuleID = 'ld3.c'
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc-unknown-linux-gnu"

define ppc_fp128 @plus(ppc_fp128 %x, ppc_fp128 %y) {
entry:
	%x_addr = alloca ppc_fp128		; <ptr> [#uses=2]
	%y_addr = alloca ppc_fp128		; <ptr> [#uses=2]
	%retval = alloca ppc_fp128, align 16		; <ptr> [#uses=2]
	%tmp = alloca ppc_fp128, align 16		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ppc_fp128 %x, ptr %x_addr
	store ppc_fp128 %y, ptr %y_addr
	%tmp1 = load ppc_fp128, ptr %x_addr, align 16		; <ppc_fp128> [#uses=1]
	%tmp2 = load ppc_fp128, ptr %y_addr, align 16		; <ppc_fp128> [#uses=1]
	%tmp3 = fadd ppc_fp128 %tmp1, %tmp2		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp3, ptr %tmp, align 16
	%tmp4 = load ppc_fp128, ptr %tmp, align 16		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp4, ptr %retval, align 16
	br label %return

return:		; preds = %entry
	%retval5 = load ppc_fp128, ptr %retval		; <ppc_fp128> [#uses=1]
	ret ppc_fp128 %retval5
}

define ppc_fp128 @minus(ppc_fp128 %x, ppc_fp128 %y) {
entry:
	%x_addr = alloca ppc_fp128		; <ptr> [#uses=2]
	%y_addr = alloca ppc_fp128		; <ptr> [#uses=2]
	%retval = alloca ppc_fp128, align 16		; <ptr> [#uses=2]
	%tmp = alloca ppc_fp128, align 16		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ppc_fp128 %x, ptr %x_addr
	store ppc_fp128 %y, ptr %y_addr
	%tmp1 = load ppc_fp128, ptr %x_addr, align 16		; <ppc_fp128> [#uses=1]
	%tmp2 = load ppc_fp128, ptr %y_addr, align 16		; <ppc_fp128> [#uses=1]
	%tmp3 = fsub ppc_fp128 %tmp1, %tmp2		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp3, ptr %tmp, align 16
	%tmp4 = load ppc_fp128, ptr %tmp, align 16		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp4, ptr %retval, align 16
	br label %return

return:		; preds = %entry
	%retval5 = load ppc_fp128, ptr %retval		; <ppc_fp128> [#uses=1]
	ret ppc_fp128 %retval5
}

define ppc_fp128 @times(ppc_fp128 %x, ppc_fp128 %y) {
entry:
	%x_addr = alloca ppc_fp128		; <ptr> [#uses=2]
	%y_addr = alloca ppc_fp128		; <ptr> [#uses=2]
	%retval = alloca ppc_fp128, align 16		; <ptr> [#uses=2]
	%tmp = alloca ppc_fp128, align 16		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ppc_fp128 %x, ptr %x_addr
	store ppc_fp128 %y, ptr %y_addr
	%tmp1 = load ppc_fp128, ptr %x_addr, align 16		; <ppc_fp128> [#uses=1]
	%tmp2 = load ppc_fp128, ptr %y_addr, align 16		; <ppc_fp128> [#uses=1]
	%tmp3 = fmul ppc_fp128 %tmp1, %tmp2		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp3, ptr %tmp, align 16
	%tmp4 = load ppc_fp128, ptr %tmp, align 16		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp4, ptr %retval, align 16
	br label %return

return:		; preds = %entry
	%retval5 = load ppc_fp128, ptr %retval		; <ppc_fp128> [#uses=1]
	ret ppc_fp128 %retval5
}

define ppc_fp128 @divide(ppc_fp128 %x, ppc_fp128 %y) {
entry:
	%x_addr = alloca ppc_fp128		; <ptr> [#uses=2]
	%y_addr = alloca ppc_fp128		; <ptr> [#uses=2]
	%retval = alloca ppc_fp128, align 16		; <ptr> [#uses=2]
	%tmp = alloca ppc_fp128, align 16		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store ppc_fp128 %x, ptr %x_addr
	store ppc_fp128 %y, ptr %y_addr
	%tmp1 = load ppc_fp128, ptr %x_addr, align 16		; <ppc_fp128> [#uses=1]
	%tmp2 = load ppc_fp128, ptr %y_addr, align 16		; <ppc_fp128> [#uses=1]
	%tmp3 = fdiv ppc_fp128 %tmp1, %tmp2		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp3, ptr %tmp, align 16
	%tmp4 = load ppc_fp128, ptr %tmp, align 16		; <ppc_fp128> [#uses=1]
	store ppc_fp128 %tmp4, ptr %retval, align 16
	br label %return

return:		; preds = %entry
	%retval5 = load ppc_fp128, ptr %retval		; <ppc_fp128> [#uses=1]
	ret ppc_fp128 %retval5
}
