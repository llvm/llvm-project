; RUN: llvm-as < %s | llc | grep powixf2
; RUN: llvm-as < %s | llc | grep fsqrt
; ModuleID = 'yyy.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define x86_fp80 @foo(x86_fp80 %x) {
entry:
	%x_addr = alloca x86_fp80		; <x86_fp80*> [#uses=2]
	%retval = alloca x86_fp80		; <x86_fp80*> [#uses=2]
	%tmp = alloca x86_fp80		; <x86_fp80*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store x86_fp80 %x, x86_fp80* %x_addr
	%tmp1 = load x86_fp80* %x_addr, align 16		; <x86_fp80> [#uses=1]
	%tmp2 = call x86_fp80 @llvm.sqrt.f80( x86_fp80 %tmp1 )		; <x86_fp80> [#uses=1]
	store x86_fp80 %tmp2, x86_fp80* %tmp, align 16
	%tmp3 = load x86_fp80* %tmp, align 16		; <x86_fp80> [#uses=1]
	store x86_fp80 %tmp3, x86_fp80* %retval, align 16
	br label %return

return:		; preds = %entry
	%retval4 = load x86_fp80* %retval		; <x86_fp80> [#uses=1]
	ret x86_fp80 %retval4
}

declare x86_fp80 @llvm.sqrt.f80(x86_fp80)

define x86_fp80 @bar(x86_fp80 %x) {
entry:
	%x_addr = alloca x86_fp80		; <x86_fp80*> [#uses=2]
	%retval = alloca x86_fp80		; <x86_fp80*> [#uses=2]
	%tmp = alloca x86_fp80		; <x86_fp80*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	store x86_fp80 %x, x86_fp80* %x_addr
	%tmp1 = load x86_fp80* %x_addr, align 16		; <x86_fp80> [#uses=1]
	%tmp2 = call x86_fp80 @llvm.powi.f80( x86_fp80 %tmp1, i32 3 )		; <x86_fp80> [#uses=1]
	store x86_fp80 %tmp2, x86_fp80* %tmp, align 16
	%tmp3 = load x86_fp80* %tmp, align 16		; <x86_fp80> [#uses=1]
	store x86_fp80 %tmp3, x86_fp80* %retval, align 16
	br label %return

return:		; preds = %entry
	%retval4 = load x86_fp80* %retval		; <x86_fp80> [#uses=1]
	ret x86_fp80 %retval4
}

declare x86_fp80 @llvm.powi.f80(x86_fp80, i32)
