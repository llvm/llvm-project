; RUN: llc < %s -march=x86 -mattr=+sse,+sse2,+sse41 | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8.7.2"

define i16 @test1(float %f) nounwind {
	%tmp = insertelement <4 x float> undef, float %f, i32 0		; <<4 x float>> [#uses=1]
	%tmp10 = insertelement <4 x float> %tmp, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	%tmp28 = tail call <4 x float> @llvm.x86.sse.sub.ss( <4 x float> %tmp12, <4 x float> < float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp37 = tail call <4 x float> @llvm.x86.sse.mul.ss( <4 x float> %tmp28, <4 x float> < float 5.000000e-01, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp48 = tail call <4 x float> @llvm.x86.sse.min.ss( <4 x float> %tmp37, <4 x float> < float 6.553500e+04, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00 > )		; <<4 x float>> [#uses=1]
	%tmp59 = tail call <4 x float> @llvm.x86.sse.max.ss( <4 x float> %tmp48, <4 x float> zeroinitializer )		; <<4 x float>> [#uses=1]
	%tmp.upgrd.1 = tail call i32 @llvm.x86.sse.cvttss2si( <4 x float> %tmp59 )		; <i32> [#uses=1]
	%tmp69 = trunc i32 %tmp.upgrd.1 to i16		; <i16> [#uses=1]
	ret i16 %tmp69
; CHECK: test1:
; CHECK: subss	LCPI0_
; CHECK: mulss	LCPI0_
; CHECK: minss	LCPI0_
}

define i16 @test2(float %f) nounwind {
	%tmp28 = fsub float %f, 1.000000e+00		; <float> [#uses=1]
	%tmp37 = fmul float %tmp28, 5.000000e-01		; <float> [#uses=1]
	%tmp375 = insertelement <4 x float> undef, float %tmp37, i32 0		; <<4 x float>> [#uses=1]
	%tmp48 = tail call <4 x float> @llvm.x86.sse.min.ss( <4 x float> %tmp375, <4 x float> < float 6.553500e+04, float undef, float undef, float undef > )		; <<4 x float>> [#uses=1]
	%tmp59 = tail call <4 x float> @llvm.x86.sse.max.ss( <4 x float> %tmp48, <4 x float> < float 0.000000e+00, float undef, float undef, float undef > )		; <<4 x float>> [#uses=1]
	%tmp = tail call i32 @llvm.x86.sse.cvttss2si( <4 x float> %tmp59 )		; <i32> [#uses=1]
	%tmp69 = trunc i32 %tmp to i16		; <i16> [#uses=1]
	ret i16 %tmp69
; CHECK: test2:
; CHECK: addss	LCPI1_
; CHECK: mulss	LCPI1_
; CHECK: minss	LCPI1_
}

declare <4 x float> @llvm.x86.sse.sub.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.mul.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.min.ss(<4 x float>, <4 x float>)

declare <4 x float> @llvm.x86.sse.max.ss(<4 x float>, <4 x float>)

declare i32 @llvm.x86.sse.cvttss2si(<4 x float>)


declare <4 x float> @llvm.x86.sse41.round.ss(<4 x float>, <4 x float>, i32)
declare <4 x float> @f()

define <4 x float> @test3(<4 x float> %A, float *%b, i32 %C) nounwind {
  %a = load float *%b
  %B = insertelement <4 x float> undef, float %a, i32 0
  %X = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %A, <4 x float> %B, i32 4)
  ret <4 x float> %X
; CHECK: test3:
; CHECK: roundss	$4, (%eax), %xmm0
}

define <4 x float> @test4(<4 x float> %A, float *%b, i32 %C) nounwind {
  %a = load float *%b
  %B = insertelement <4 x float> undef, float %a, i32 0
  %q = call <4 x float> @f()
  %X = call <4 x float> @llvm.x86.sse41.round.ss(<4 x float> %q, <4 x float> %B, i32 4)
  ret <4 x float> %X
; CHECK: test4:
; CHECK: movss	(%eax), %xmm
; CHECK: call
; CHECK: roundss $4, %xmm{{.*}}, %xmm0
}

; PR13576 
define  <2 x double> @test5() nounwind uwtable readnone noinline {
entry:
  %0 = tail call <2 x double> @llvm.x86.sse2.cvtsi2sd(<2 x double> <double
4.569870e+02, double 1.233210e+02>, i32 128) nounwind readnone
  ret <2 x double> %0
; CHECK: test5:
; CHECK: mov
; CHECK: mov
; CHECK: cvtsi2sd
}

declare <2 x double> @llvm.x86.sse2.cvtsi2sd(<2 x double>, i32) nounwind readnone
