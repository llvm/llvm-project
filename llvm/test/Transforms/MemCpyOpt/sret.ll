; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | not grep {call.*memcpy}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9"

define void @ccosl({ x86_fp80, x86_fp80 }* noalias sret  %agg.result, { x86_fp80, x86_fp80 }* byval  %z) nounwind  {
entry:
	%iz = alloca { x86_fp80, x86_fp80 }		; <{ x86_fp80, x86_fp80 }*> [#uses=3]
	%memtmp = alloca { x86_fp80, x86_fp80 }, align 16		; <{ x86_fp80, x86_fp80 }*> [#uses=2]
	%tmp1 = getelementptr { x86_fp80, x86_fp80 }* %z, i32 0, i32 1		; <x86_fp80*> [#uses=1]
	%tmp2 = load x86_fp80* %tmp1, align 16		; <x86_fp80> [#uses=1]
	%tmp3 = sub x86_fp80 0xK80000000000000000000, %tmp2		; <x86_fp80> [#uses=1]
	%tmp4 = getelementptr { x86_fp80, x86_fp80 }* %iz, i32 0, i32 1		; <x86_fp80*> [#uses=1]
	%real = getelementptr { x86_fp80, x86_fp80 }* %iz, i32 0, i32 0		; <x86_fp80*> [#uses=1]
	%tmp7 = getelementptr { x86_fp80, x86_fp80 }* %z, i32 0, i32 0		; <x86_fp80*> [#uses=1]
	%tmp8 = load x86_fp80* %tmp7, align 16		; <x86_fp80> [#uses=1]
	store x86_fp80 %tmp3, x86_fp80* %real, align 16
	store x86_fp80 %tmp8, x86_fp80* %tmp4, align 16
	call void @ccoshl( { x86_fp80, x86_fp80 }* noalias sret  %memtmp, { x86_fp80, x86_fp80 }* byval  %iz ) nounwind 
	%memtmp14 = bitcast { x86_fp80, x86_fp80 }* %memtmp to i8*		; <i8*> [#uses=1]
	%agg.result15 = bitcast { x86_fp80, x86_fp80 }* %agg.result to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %agg.result15, i8* %memtmp14, i32 32, i32 16 )
	ret void
}

declare void @ccoshl({ x86_fp80, x86_fp80 }* noalias sret , { x86_fp80, x86_fp80 }* byval ) nounwind 

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 
