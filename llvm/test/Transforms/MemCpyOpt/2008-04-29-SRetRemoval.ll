; RUN: llvm-as < %s | opt -memcpyopt | llvm-dis | grep {call.*memcpy.*agg.result}

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"
@x = external global { x86_fp80, x86_fp80 }		; <{ x86_fp80, x86_fp80 }*> [#uses=1]

define void @foo({ x86_fp80, x86_fp80 }* noalias sret %agg.result) nounwind  {
entry:
	%x.0 = alloca { x86_fp80, x86_fp80 }		; <{ x86_fp80, x86_fp80 }*> [#uses=1]
	%x.01 = bitcast { x86_fp80, x86_fp80 }* %x.0 to i8*		; <i8*> [#uses=2]
	call void @llvm.memcpy.i32( i8* %x.01, i8* bitcast ({ x86_fp80, x86_fp80 }* @x to i8*), i32 32, i32 16 )
	%agg.result2 = bitcast { x86_fp80, x86_fp80 }* %agg.result to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %agg.result2, i8* %x.01, i32 32, i32 16 )
	ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 
