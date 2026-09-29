; RUN: llc < %s -mattr=+sse4.1
; rdar://5886601
; gcc testsuite:  gcc.target/i386/sse4_1-pblendw.c
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin8"

define i32 @main() nounwind  {
entry:
	%tmp122 = load <2 x i64>, ptr null, align 16		; <<2 x i64>> [#uses=1]
	%tmp126 = bitcast <2 x i64> %tmp122 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp129 = call <8 x i16> @llvm.x86.sse41.pblendw( <8 x i16> zeroinitializer, <8 x i16> %tmp126, i32 2 ) nounwind 		; <<8 x i16>> [#uses=0]
	ret i32 0
}

declare <8 x i16> @llvm.x86.sse41.pblendw(<8 x i16>, <8 x i16>, i32) nounwind 
