; RUN: llvm-as < %s | llc -march=x86 -mattr=sse41 -o %t -f
; RUN: grep unpcklps %t | count 1
; RUN: grep unpckhps %t | count 3

; Transpose example using the more generic vector shuffle. Return float8
; instead of float16
; ModuleID = 'transpose2_opt.bc'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-apple-cl.1.0"
@r0 = common global <4 x float> zeroinitializer, align 16		; <<4 x float>*> [#uses=1]
@r1 = common global <4 x float> zeroinitializer, align 16		; <<4 x float>*> [#uses=1]
@r2 = common global <4 x float> zeroinitializer, align 16		; <<4 x float>*> [#uses=1]
@r3 = common global <4 x float> zeroinitializer, align 16		; <<4 x float>*> [#uses=1]

define <8 x float> @__transpose2(<4 x float> %p0, <4 x float> %p1, <4 x float> %p2, <4 x float> %p3) nounwind {
entry:
	%unpcklps = shufflevector <4 x float> %p0, <4 x float> %p2, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>> [#uses=2]
	%unpckhps = shufflevector <4 x float> %p0, <4 x float> %p2, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=2]
	%unpcklps8 = shufflevector <4 x float> %p1, <4 x float> %p3, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>> [#uses=2]
	%unpckhps11 = shufflevector <4 x float> %p1, <4 x float> %p3, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=2]
	%unpcklps14 = shufflevector <4 x float> %unpcklps, <4 x float> %unpcklps8, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>> [#uses=1]
	%unpckhps17 = shufflevector <4 x float> %unpcklps, <4 x float> %unpcklps8, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=1]
        %r1 = shufflevector <4 x float> %unpcklps14,  <4 x float> %unpckhps17,  <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >; 
	%unpcklps20 = shufflevector <4 x float> %unpckhps, <4 x float> %unpckhps11, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>> [#uses=1]
	%unpckhps23 = shufflevector <4 x float> %unpckhps, <4 x float> %unpckhps11, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=1]
        %r2 = shufflevector <4 x float> %unpcklps20,  <4 x float> %unpckhps23,  <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >;
;       %r3 = shufflevector <8 x float> %r1,  <8 x float> %r2,  <16 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15 >; 
	ret <8 x float> %r2;
}
