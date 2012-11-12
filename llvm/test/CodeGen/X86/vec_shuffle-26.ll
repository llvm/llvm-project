; RUN: llc < %s -march=x86 -mcpu=generic -mattr=sse41 | FileCheck %s
; RUN: llc < %s -march=x86 -mcpu=atom | FileCheck -check-prefix=ATOM %s

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
; CHECK: transpose2
; CHECK: unpckhps
; CHECK: unpckhps
; CHECK: unpcklps
; CHECK: unpckhps
; Different instruction order for Atom.
; ATOM: transpose2
; ATOM: unpckhps
; ATOM: unpckhps
; ATOM: unpckhps
; ATOM: unpcklps
	%unpcklps = shufflevector <4 x float> %p0, <4 x float> %p2, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>> [#uses=2]
	%unpckhps = shufflevector <4 x float> %p0, <4 x float> %p2, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=2]
	%unpcklps8 = shufflevector <4 x float> %p1, <4 x float> %p3, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>> [#uses=2]
	%unpckhps11 = shufflevector <4 x float> %p1, <4 x float> %p3, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=2]
	%unpcklps14 = shufflevector <4 x float> %unpcklps, <4 x float> %unpcklps8, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>> [#uses=1]
	%unpckhps17 = shufflevector <4 x float> %unpcklps, <4 x float> %unpcklps8, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=1]
        %r1 = shufflevector <4 x float> %unpcklps14,  <4 x float> %unpckhps17,  <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
	%unpcklps20 = shufflevector <4 x float> %unpckhps, <4 x float> %unpckhps11, <4 x i32> < i32 0, i32 4, i32 1, i32 5 >		; <<4 x float>> [#uses=1]
	%unpckhps23 = shufflevector <4 x float> %unpckhps, <4 x float> %unpckhps11, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=1]
        %r2 = shufflevector <4 x float> %unpcklps20,  <4 x float> %unpckhps23,  <8 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
;       %r3 = shufflevector <8 x float> %r1,  <8 x float> %r2,  <16 x i32> < i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15 >; 
	ret <8 x float> %r2
}

define <2 x i64> @lo_hi_shift(float* nocapture %x, float* nocapture %y) nounwind {
entry:
; movhps should happen before extractps to assure it gets the correct value.
; CHECK: lo_hi_shift
; CHECK: movhps ([[BASEREG:%[a-z]+]]),
; CHECK: extractps ${{[0-9]+}}, %xmm{{[0-9]+}}, {{[0-9]*}}([[BASEREG]])
; CHECK: extractps ${{[0-9]+}}, %xmm{{[0-9]+}}, {{[0-9]*}}([[BASEREG]])
; ATOM: lo_hi_shift
; ATOM: movhps ([[BASEREG:%[a-z]+]]),
; ATOM: movd %xmm{{[0-9]+}}, {{[0-9]*}}([[BASEREG]])
; ATOM: movd %xmm{{[0-9]+}}, {{[0-9]*}}([[BASEREG]])
  %v.i = bitcast float* %y to <4 x float>*
  %0 = load <4 x float>* %v.i, align 1
  %1 = bitcast float* %x to <1 x i64>*
  %.val = load <1 x i64>* %1, align 1
  %2 = bitcast <1 x i64> %.val to <2 x float>
  %shuffle.i = shufflevector <2 x float> %2, <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
  %shuffle1.i = shufflevector <4 x float> %0, <4 x float> %shuffle.i, <4 x i32> <i32 0, i32 1, i32 4, i32 5>
  %cast.i = bitcast <4 x float> %0 to <2 x i64>
  %extract.i = extractelement <2 x i64> %cast.i, i32 1
  %3 = bitcast float* %x to i64*
  store i64 %extract.i, i64* %3, align 4
  %4 = bitcast <4 x float> %0 to <16 x i8>
  %5 = bitcast <4 x float> %shuffle1.i to <16 x i8>
  %palignr = shufflevector <16 x i8> %5, <16 x i8> %4, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  %6 = bitcast <16 x i8> %palignr to <2 x i64>
  ret <2 x i64> %6
}
