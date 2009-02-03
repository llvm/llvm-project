; RUN: llvm-as < %s | opt -scalarrepl | llvm-dis | grep {call.*mem} 
; PR2369

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define void @memtest1(i8* %dst, i8* %src) nounwind  {
entry:
	%temp = alloca [100 x i8]		; <[100 x i8]*> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%temp1 = bitcast [100 x i8]* %temp to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %temp1, i8* %src, i32 100, i32 1 )
	%temp3 = bitcast [100 x i8]* %temp to i8*		; <i8*> [#uses=1]
	call void @llvm.memcpy.i32( i8* %dst, i8* %temp3, i32 100, i32 1 )
	ret void
}

declare void @llvm.memcpy.i32(i8*, i8*, i32, i32) nounwind 
