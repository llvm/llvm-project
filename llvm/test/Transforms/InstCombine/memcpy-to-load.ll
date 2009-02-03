; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep {load double}
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i686-apple-darwin8"

define void @foo(double* %X, double* %Y) {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp2 = bitcast double* %X to i8*		; <i8*> [#uses=1]
	%tmp13 = bitcast double* %Y to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %tmp2, i8* %tmp13, i32 8, i32 1 )
	ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 
