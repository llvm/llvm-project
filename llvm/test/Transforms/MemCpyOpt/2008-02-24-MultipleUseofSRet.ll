; RUN: llvm-as < %s | opt -memcpyopt -dse | llvm-dis | grep {call.*initialize} | not grep memtmp
; PR2077

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"

define internal fastcc void @initialize({ x86_fp80, x86_fp80 }* noalias sret  %agg.result) nounwind  {
entry:
	%agg.result.03 = getelementptr { x86_fp80, x86_fp80 }* %agg.result, i32 0, i32 0		; <x86_fp80*> [#uses=1]
	store x86_fp80 0xK00000000000000000000, x86_fp80* %agg.result.03
	%agg.result.15 = getelementptr { x86_fp80, x86_fp80 }* %agg.result, i32 0, i32 1		; <x86_fp80*> [#uses=1]
	store x86_fp80 0xK00000000000000000000, x86_fp80* %agg.result.15
	ret void
}

declare fastcc x86_fp80 @passed_uninitialized({ x86_fp80, x86_fp80 }* %x) nounwind

define fastcc void @badly_optimized() nounwind  {
entry:
	%z = alloca { x86_fp80, x86_fp80 }		; <{ x86_fp80, x86_fp80 }*> [#uses=2]
	%tmp = alloca { x86_fp80, x86_fp80 }		; <{ x86_fp80, x86_fp80 }*> [#uses=2]
	%memtmp = alloca { x86_fp80, x86_fp80 }, align 8		; <{ x86_fp80, x86_fp80 }*> [#uses=2]
	call fastcc void @initialize( { x86_fp80, x86_fp80 }* noalias sret  %memtmp )
	%tmp1 = bitcast { x86_fp80, x86_fp80 }* %tmp to i8*		; <i8*> [#uses=1]
	%memtmp2 = bitcast { x86_fp80, x86_fp80 }* %memtmp to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %tmp1, i8* %memtmp2, i32 24, i32 8 )
	%z3 = bitcast { x86_fp80, x86_fp80 }* %z to i8*		; <i8*> [#uses=1]
	%tmp4 = bitcast { x86_fp80, x86_fp80 }* %tmp to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %z3, i8* %tmp4, i32 24, i32 8 )
	%tmp5 = call fastcc x86_fp80 @passed_uninitialized( { x86_fp80, x86_fp80 }* %z )		; <x86_fp80> [#uses=0]
	ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind
