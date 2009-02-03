; RUN: llvm-as < %s | opt -memcpyopt -dse | llvm-dis | grep {call.*memcpy} | count 1

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin9"

define void @ccosl({ x86_fp80, x86_fp80 }* sret  %agg.result, x86_fp80 %z.0, x86_fp80 %z.1) nounwind  {
entry:
	%tmp2 = alloca { x86_fp80, x86_fp80 }		; <{ x86_fp80, x86_fp80 }*> [#uses=1]
	%memtmp = alloca { x86_fp80, x86_fp80 }, align 16		; <{ x86_fp80, x86_fp80 }*> [#uses=2]
	%tmp5 = sub x86_fp80 0xK80000000000000000000, %z.1		; <x86_fp80> [#uses=1]
	call void @ccoshl( { x86_fp80, x86_fp80 }* sret  %memtmp, x86_fp80 %tmp5, x86_fp80 %z.0 ) nounwind 
	%tmp219 = bitcast { x86_fp80, x86_fp80 }* %tmp2 to i8*		; <i8*> [#uses=2]
	%memtmp20 = bitcast { x86_fp80, x86_fp80 }* %memtmp to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %tmp219, i8* %memtmp20, i32 32, i32 16 )
	%agg.result21 = bitcast { x86_fp80, x86_fp80 }* %agg.result to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %agg.result21, i8* %tmp219, i32 32, i32 16 )
	ret void
}

declare void @ccoshl({ x86_fp80, x86_fp80 }* sret , x86_fp80, x86_fp80) nounwind 

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 
