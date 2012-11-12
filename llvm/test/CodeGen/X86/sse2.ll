; Tests for SSE2 and below, without SSE3+.
; RUN: llc < %s -mtriple=i386-apple-darwin10 -mcpu=pentium4 -O3 | FileCheck %s

define void @test1(<2 x double>* %r, <2 x double>* %A, double %B) nounwind  {
	%tmp3 = load <2 x double>* %A, align 16
	%tmp7 = insertelement <2 x double> undef, double %B, i32 0
	%tmp9 = shufflevector <2 x double> %tmp3, <2 x double> %tmp7, <2 x i32> < i32 2, i32 1 >
	store <2 x double> %tmp9, <2 x double>* %r, align 16
	ret void
        
; CHECK: test1:
; CHECK: 	movl	8(%esp), %eax
; CHECK-NEXT: 	movapd	(%eax), %xmm0
; CHECK-NEXT: 	movlpd	12(%esp), %xmm0
; CHECK-NEXT: 	movl	4(%esp), %eax
; CHECK-NEXT: 	movapd	%xmm0, (%eax)
; CHECK-NEXT: 	ret
}

define void @test2(<2 x double>* %r, <2 x double>* %A, double %B) nounwind  {
	%tmp3 = load <2 x double>* %A, align 16
	%tmp7 = insertelement <2 x double> undef, double %B, i32 0
	%tmp9 = shufflevector <2 x double> %tmp3, <2 x double> %tmp7, <2 x i32> < i32 0, i32 2 >
	store <2 x double> %tmp9, <2 x double>* %r, align 16
	ret void
        
; CHECK: test2:
; CHECK: 	movl	8(%esp), %eax
; CHECK-NEXT: 	movapd	(%eax), %xmm0
; CHECK-NEXT: 	movhpd	12(%esp), %xmm0
; CHECK-NEXT: 	movl	4(%esp), %eax
; CHECK-NEXT: 	movapd	%xmm0, (%eax)
; CHECK-NEXT: 	ret
}


define void @test3(<4 x float>* %res, <4 x float>* %A, <4 x float>* %B) nounwind {
	%tmp = load <4 x float>* %B		; <<4 x float>> [#uses=2]
	%tmp3 = load <4 x float>* %A		; <<4 x float>> [#uses=2]
	%tmp.upgrd.1 = extractelement <4 x float> %tmp3, i32 0		; <float> [#uses=1]
	%tmp7 = extractelement <4 x float> %tmp, i32 0		; <float> [#uses=1]
	%tmp8 = extractelement <4 x float> %tmp3, i32 1		; <float> [#uses=1]
	%tmp9 = extractelement <4 x float> %tmp, i32 1		; <float> [#uses=1]
	%tmp10 = insertelement <4 x float> undef, float %tmp.upgrd.1, i32 0		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float %tmp7, i32 1		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float %tmp8, i32 2		; <<4 x float>> [#uses=1]
	%tmp13 = insertelement <4 x float> %tmp12, float %tmp9, i32 3		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp13, <4 x float>* %res
	ret void
; CHECK: @test3
; CHECK: 	unpcklps	
}

define void @test4(<4 x float> %X, <4 x float>* %res) nounwind {
	%tmp5 = shufflevector <4 x float> %X, <4 x float> undef, <4 x i32> < i32 2, i32 6, i32 3, i32 7 >		; <<4 x float>> [#uses=1]
	store <4 x float> %tmp5, <4 x float>* %res
	ret void
; CHECK: @test4
; CHECK: 	pshufd	$50, %xmm0, %xmm0
}

define <4 x i32> @test5(i8** %ptr) nounwind {
; CHECK: test5:
; CHECK: pxor
; CHECK: punpcklbw
; CHECK: punpcklwd

	%tmp = load i8** %ptr		; <i8*> [#uses=1]
	%tmp.upgrd.1 = bitcast i8* %tmp to float*		; <float*> [#uses=1]
	%tmp.upgrd.2 = load float* %tmp.upgrd.1		; <float> [#uses=1]
	%tmp.upgrd.3 = insertelement <4 x float> undef, float %tmp.upgrd.2, i32 0		; <<4 x float>> [#uses=1]
	%tmp9 = insertelement <4 x float> %tmp.upgrd.3, float 0.000000e+00, i32 1		; <<4 x float>> [#uses=1]
	%tmp10 = insertelement <4 x float> %tmp9, float 0.000000e+00, i32 2		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp10, float 0.000000e+00, i32 3		; <<4 x float>> [#uses=1]
	%tmp21 = bitcast <4 x float> %tmp11 to <16 x i8>		; <<16 x i8>> [#uses=1]
	%tmp22 = shufflevector <16 x i8> %tmp21, <16 x i8> zeroinitializer, <16 x i32> < i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23 >		; <<16 x i8>> [#uses=1]
	%tmp31 = bitcast <16 x i8> %tmp22 to <8 x i16>		; <<8 x i16>> [#uses=1]
	%tmp.upgrd.4 = shufflevector <8 x i16> zeroinitializer, <8 x i16> %tmp31, <8 x i32> < i32 0, i32 8, i32 1, i32 9, i32 2, i32 10, i32 3, i32 11 >		; <<8 x i16>> [#uses=1]
	%tmp36 = bitcast <8 x i16> %tmp.upgrd.4 to <4 x i32>		; <<4 x i32>> [#uses=1]
	ret <4 x i32> %tmp36
}

define void @test6(<4 x float>* %res, <4 x float>* %A) nounwind {
        %tmp1 = load <4 x float>* %A            ; <<4 x float>> [#uses=1]
        %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> < i32 0, i32 5, i32 6, i32 7 >          ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp2, <4 x float>* %res
        ret void
        
; CHECK: test6:
; CHECK: 	movaps	(%eax), %xmm0
; CHECK:	movaps	%xmm0, (%eax)
}

define void @test7() nounwind {
        bitcast <4 x i32> zeroinitializer to <4 x float>                ; <<4 x float>>:1 [#uses=1]
        shufflevector <4 x float> %1, <4 x float> zeroinitializer, <4 x i32> zeroinitializer         ; <<4 x float>>:2 [#uses=1]
        store <4 x float> %2, <4 x float>* null
        ret void
        
; CHECK: test7:
; CHECK:	xorps	%xmm0, %xmm0
; CHECK:	movaps	%xmm0, 0
}

@x = external global [4 x i32]

define <2 x i64> @test8() nounwind {
	%tmp = load i32* getelementptr ([4 x i32]* @x, i32 0, i32 0)		; <i32> [#uses=1]
	%tmp3 = load i32* getelementptr ([4 x i32]* @x, i32 0, i32 1)		; <i32> [#uses=1]
	%tmp5 = load i32* getelementptr ([4 x i32]* @x, i32 0, i32 2)		; <i32> [#uses=1]
	%tmp7 = load i32* getelementptr ([4 x i32]* @x, i32 0, i32 3)		; <i32> [#uses=1]
	%tmp.upgrd.1 = insertelement <4 x i32> undef, i32 %tmp, i32 0		; <<4 x i32>> [#uses=1]
	%tmp13 = insertelement <4 x i32> %tmp.upgrd.1, i32 %tmp3, i32 1		; <<4 x i32>> [#uses=1]
	%tmp14 = insertelement <4 x i32> %tmp13, i32 %tmp5, i32 2		; <<4 x i32>> [#uses=1]
	%tmp15 = insertelement <4 x i32> %tmp14, i32 %tmp7, i32 3		; <<4 x i32>> [#uses=1]
	%tmp16 = bitcast <4 x i32> %tmp15 to <2 x i64>		; <<2 x i64>> [#uses=1]
	ret <2 x i64> %tmp16
; CHECK: test8:
; CHECK: movups	(%eax), %xmm0
}

define <4 x float> @test9(i32 %dummy, float %a, float %b, float %c, float %d) nounwind {
	%tmp = insertelement <4 x float> undef, float %a, i32 0		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp, float %b, i32 1		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float %c, i32 2		; <<4 x float>> [#uses=1]
	%tmp13 = insertelement <4 x float> %tmp12, float %d, i32 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp13
; CHECK: test9:
; CHECK: movups	8(%esp), %xmm0
}

define <4 x float> @test10(float %a, float %b, float %c, float %d) nounwind {
	%tmp = insertelement <4 x float> undef, float %a, i32 0		; <<4 x float>> [#uses=1]
	%tmp11 = insertelement <4 x float> %tmp, float %b, i32 1		; <<4 x float>> [#uses=1]
	%tmp12 = insertelement <4 x float> %tmp11, float %c, i32 2		; <<4 x float>> [#uses=1]
	%tmp13 = insertelement <4 x float> %tmp12, float %d, i32 3		; <<4 x float>> [#uses=1]
	ret <4 x float> %tmp13
; CHECK: test10:
; CHECK: movaps	4(%esp), %xmm0
}

define <2 x double> @test11(double %a, double %b) nounwind {
	%tmp = insertelement <2 x double> undef, double %a, i32 0		; <<2 x double>> [#uses=1]
	%tmp7 = insertelement <2 x double> %tmp, double %b, i32 1		; <<2 x double>> [#uses=1]
	ret <2 x double> %tmp7
; CHECK: test11:
; CHECK: movaps	4(%esp), %xmm0
}

define void @test12() nounwind {
        %tmp1 = load <4 x float>* null          ; <<4 x float>> [#uses=2]
        %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> < float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00 >, <4 x i32> < i32 0, i32 1, i32 6, i32 7 >             ; <<4 x float>> [#uses=1]
        %tmp3 = shufflevector <4 x float> %tmp1, <4 x float> zeroinitializer, <4 x i32> < i32 2, i32 3, i32 6, i32 7 >                ; <<4 x float>> [#uses=1]
        %tmp4 = fadd <4 x float> %tmp2, %tmp3            ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp4, <4 x float>* null
        ret void
; CHECK: test12:
; CHECK: movhlps
; CHECK: shufps
}

define void @test13(<4 x float>* %res, <4 x float>* %A, <4 x float>* %B, <4 x float>* %C) nounwind {
        %tmp3 = load <4 x float>* %B            ; <<4 x float>> [#uses=1]
        %tmp5 = load <4 x float>* %C            ; <<4 x float>> [#uses=1]
        %tmp11 = shufflevector <4 x float> %tmp3, <4 x float> %tmp5, <4 x i32> < i32 1, i32 4, i32 1, i32 5 >         ; <<4 x float>> [#uses=1]
        store <4 x float> %tmp11, <4 x float>* %res
        ret void
; CHECK: test13
; CHECK: shufps	$69, (%eax), %xmm0
; CHECK: pshufd	$-40, %xmm0, %xmm0
}

define <4 x float> @test14(<4 x float>* %x, <4 x float>* %y) nounwind {
        %tmp = load <4 x float>* %y             ; <<4 x float>> [#uses=2]
        %tmp5 = load <4 x float>* %x            ; <<4 x float>> [#uses=2]
        %tmp9 = fadd <4 x float> %tmp5, %tmp             ; <<4 x float>> [#uses=1]
        %tmp21 = fsub <4 x float> %tmp5, %tmp            ; <<4 x float>> [#uses=1]
        %tmp27 = shufflevector <4 x float> %tmp9, <4 x float> %tmp21, <4 x i32> < i32 0, i32 1, i32 4, i32 5 >                ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp27
; CHECK: test14:
; CHECK: 	subps	[[X1:%xmm[0-9]+]], [[X2:%xmm[0-9]+]]
; CHECK: 	addps	[[X1]], [[X0:%xmm[0-9]+]]
; CHECK: 	movlhps	[[X2]], [[X0]]
}

define <4 x float> @test15(<4 x float>* %x, <4 x float>* %y) nounwind {
entry:
        %tmp = load <4 x float>* %y             ; <<4 x float>> [#uses=1]
        %tmp3 = load <4 x float>* %x            ; <<4 x float>> [#uses=1]
        %tmp4 = shufflevector <4 x float> %tmp3, <4 x float> %tmp, <4 x i32> < i32 2, i32 3, i32 6, i32 7 >           ; <<4 x float>> [#uses=1]
        ret <4 x float> %tmp4
; CHECK: test15:
; CHECK: 	movhlps	%xmm1, %xmm0
}

; PR8900
; CHECK: test16:
; CHECK: unpcklpd
; CHECK: ret

define  <2 x double> @test16(<4 x double> * nocapture %srcA, <2 x double>* nocapture %dst) {
  %i5 = getelementptr inbounds <4 x double>* %srcA, i32 3
  %i6 = load <4 x double>* %i5, align 32
  %i7 = shufflevector <4 x double> %i6, <4 x double> undef, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %i7
}

; PR9009
define fastcc void @test17() nounwind {
entry:
  %0 = insertelement <4 x i32> undef, i32 undef, i32 1
  %1 = shufflevector <4 x i32> <i32 undef, i32 undef, i32 32768, i32 32768>, <4 x i32> %0, <4 x i32> <i32 4, i32 5, i32 2, i32 3>
  %2 = bitcast <4 x i32> %1 to <4 x float>
  store <4 x float> %2, <4 x float> * undef
  ret void
}

; PR9210
define <4 x float> @f(<4 x double>) nounwind {
entry:
 %double2float.i = fptrunc <4 x double> %0 to <4 x float>
 ret <4 x float> %double2float.i
}

