; RUN: opt < %s -scalarrepl -S > %t
; RUN: grep "ret <16 x float> %A" %t
; RUN: grep "ret <16 x float> zeroinitializer" %t
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"

define <16 x float> @foo(<16 x float> %A) nounwind {
	%tmp = alloca <16 x float>, align 16
	%tmp2 = alloca <16 x float>, align 16
	store <16 x float> %A, <16 x float>* %tmp
	%s = bitcast <16 x float>* %tmp to i8*
	%s2 = bitcast <16 x float>* %tmp2 to i8*
	call void @llvm.memcpy.i64(i8* %s2, i8* %s, i64 64, i32 16)
	
	%R = load <16 x float>* %tmp2
	ret <16 x float> %R
}

define <16 x float> @foo2(<16 x float> %A) nounwind {
	%tmp2 = alloca <16 x float>, align 16

	%s2 = bitcast <16 x float>* %tmp2 to i8*
	call void @llvm.memset.i64(i8* %s2, i8 0, i64 64, i32 16)
	
	%R = load <16 x float>* %tmp2
	ret <16 x float> %R
}


declare void @llvm.memcpy.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind
declare void @llvm.memset.i64(i8* nocapture, i8, i64, i32) nounwind
