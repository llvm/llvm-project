; These are tests for SSE3 codegen.

; RUN: llc < %s -march=x86-64 -mcpu=nocona -mtriple=i686-apple-darwin9 -O3 \
; RUN:              | FileCheck %s --check-prefix=X64

; Test for v8xi16 lowering where we extract the first element of the vector and
; placed it in the second element of the result.

define void @t0(<8 x i16>* %dest, <8 x i16>* %old) nounwind {
entry:
	%tmp3 = load <8 x i16>* %old
	%tmp6 = shufflevector <8 x i16> %tmp3,
                <8 x i16> < i16 0, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef, i16 undef >,
                <8 x i32> < i32 8, i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef  >
	store <8 x i16> %tmp6, <8 x i16>* %dest
	ret void
        
; X64: t0:
; X64:	movdqa	(%rsi), %xmm0
; X64:	pslldq	$2, %xmm0
; X64:	movdqa	%xmm0, (%rdi)
; X64:	ret
}

define <8 x i16> @t1(<8 x i16>* %A, <8 x i16>* %B) nounwind {
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> < i32 8, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
	ret <8 x i16> %tmp3
        
; X64: t1:
; X64: 	movdqa	(%rdi), %xmm0
; X64: 	pinsrw	$0, (%rsi), %xmm0
; X64: 	ret
}

define <8 x i16> @t2(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 9, i32 1, i32 2, i32 9, i32 4, i32 5, i32 6, i32 7 >
	ret <8 x i16> %tmp
; X64: t2:
; X64:	pextrw	$1, %xmm1, %eax
; X64:	pinsrw	$0, %eax, %xmm0
; X64:	pinsrw	$3, %eax, %xmm0
; X64:	ret
}

define <8 x i16> @t3(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %A, <8 x i32> < i32 8, i32 3, i32 2, i32 13, i32 7, i32 6, i32 5, i32 4 >
	ret <8 x i16> %tmp
; X64: t3:
; X64: 	pextrw	$5, %xmm0, %eax
; X64: 	pshuflw	$44, %xmm0, %xmm0
; X64: 	pshufhw	$27, %xmm0, %xmm0
; X64: 	pinsrw	$3, %eax, %xmm0
; X64: 	ret
}

define <8 x i16> @t4(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 0, i32 7, i32 2, i32 3, i32 1, i32 5, i32 6, i32 5 >
	ret <8 x i16> %tmp
; X64: t4:
; X64: 	pextrw	$7, [[XMM0:%xmm[0-9]+]], %eax
; X64: 	pshufhw	$100, [[XMM0]], [[XMM1:%xmm[0-9]+]]
; X64: 	pinsrw	$1, %eax, [[XMM1]]
; X64: 	pextrw	$1, [[XMM0]], %eax
; X64: 	pinsrw	$4, %eax, %xmm0
; X64: 	ret
}

define <8 x i16> @t5(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 8, i32 9, i32 0, i32 1, i32 10, i32 11, i32 2, i32 3 >
	ret <8 x i16> %tmp
; X64: 	t5:
; X64: 		movlhps	%xmm1, %xmm0
; X64: 		pshufd	$114, %xmm0, %xmm0
; X64: 		ret
}

define <8 x i16> @t6(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 8, i32 9, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7 >
	ret <8 x i16> %tmp
; X64: 	t6:
; X64: 		movss	%xmm1, %xmm0
; X64: 		ret
}

define <8 x i16> @t7(<8 x i16> %A, <8 x i16> %B) nounwind {
	%tmp = shufflevector <8 x i16> %A, <8 x i16> %B, <8 x i32> < i32 0, i32 0, i32 3, i32 2, i32 4, i32 6, i32 4, i32 7 >
	ret <8 x i16> %tmp
; X64: 	t7:
; X64: 		pshuflw	$-80, %xmm0, %xmm0
; X64: 		pshufhw	$-56, %xmm0, %xmm0
; X64: 		ret
}

define void @t8(<2 x i64>* %res, <2 x i64>* %A) nounwind {
	%tmp = load <2 x i64>* %A
	%tmp.upgrd.1 = bitcast <2 x i64> %tmp to <8 x i16>
	%tmp0 = extractelement <8 x i16> %tmp.upgrd.1, i32 0
	%tmp1 = extractelement <8 x i16> %tmp.upgrd.1, i32 1
	%tmp2 = extractelement <8 x i16> %tmp.upgrd.1, i32 2
	%tmp3 = extractelement <8 x i16> %tmp.upgrd.1, i32 3
	%tmp4 = extractelement <8 x i16> %tmp.upgrd.1, i32 4
	%tmp5 = extractelement <8 x i16> %tmp.upgrd.1, i32 5
	%tmp6 = extractelement <8 x i16> %tmp.upgrd.1, i32 6
	%tmp7 = extractelement <8 x i16> %tmp.upgrd.1, i32 7
	%tmp8 = insertelement <8 x i16> undef, i16 %tmp2, i32 0
	%tmp9 = insertelement <8 x i16> %tmp8, i16 %tmp1, i32 1
	%tmp10 = insertelement <8 x i16> %tmp9, i16 %tmp0, i32 2
	%tmp11 = insertelement <8 x i16> %tmp10, i16 %tmp3, i32 3
	%tmp12 = insertelement <8 x i16> %tmp11, i16 %tmp6, i32 4
	%tmp13 = insertelement <8 x i16> %tmp12, i16 %tmp5, i32 5
	%tmp14 = insertelement <8 x i16> %tmp13, i16 %tmp4, i32 6
	%tmp15 = insertelement <8 x i16> %tmp14, i16 %tmp7, i32 7
	%tmp15.upgrd.2 = bitcast <8 x i16> %tmp15 to <2 x i64>
	store <2 x i64> %tmp15.upgrd.2, <2 x i64>* %res
	ret void
; X64: 	t8:
; X64: 		pshuflw	$-58, (%rsi), %xmm0
; X64: 		pshufhw	$-58, %xmm0, %xmm0
; X64: 		movdqa	%xmm0, (%rdi)
; X64: 		ret
}

define void @t9(<4 x float>* %r, <2 x i32>* %A) nounwind {
	%tmp = load <4 x float>* %r
	%tmp.upgrd.3 = bitcast <2 x i32>* %A to double*
	%tmp.upgrd.4 = load double* %tmp.upgrd.3
	%tmp.upgrd.5 = insertelement <2 x double> undef, double %tmp.upgrd.4, i32 0
	%tmp5 = insertelement <2 x double> %tmp.upgrd.5, double undef, i32 1	
	%tmp6 = bitcast <2 x double> %tmp5 to <4 x float>	
	%tmp.upgrd.6 = extractelement <4 x float> %tmp, i32 0	
	%tmp7 = extractelement <4 x float> %tmp, i32 1		
	%tmp8 = extractelement <4 x float> %tmp6, i32 0		
	%tmp9 = extractelement <4 x float> %tmp6, i32 1		
	%tmp10 = insertelement <4 x float> undef, float %tmp.upgrd.6, i32 0	
	%tmp11 = insertelement <4 x float> %tmp10, float %tmp7, i32 1
	%tmp12 = insertelement <4 x float> %tmp11, float %tmp8, i32 2
	%tmp13 = insertelement <4 x float> %tmp12, float %tmp9, i32 3
	store <4 x float> %tmp13, <4 x float>* %r
	ret void
; X64: 	t9:
; X64: 		movaps	(%rdi), %xmm0
; X64:	        movhps	(%rsi), %xmm0
; X64:	        movaps	%xmm0, (%rdi)
; X64: 		ret
}



; FIXME: This testcase produces icky code. It can be made much better!
; PR2585

@g1 = external constant <4 x i32>
@g2 = external constant <4 x i16>

define internal void @t10() nounwind {
        load <4 x i32>* @g1, align 16 
        bitcast <4 x i32> %1 to <8 x i16>
        shufflevector <8 x i16> %2, <8 x i16> undef, <8 x i32> < i32 0, i32 2, i32 4, i32 6, i32 undef, i32 undef, i32 undef, i32 undef >
        bitcast <8 x i16> %3 to <2 x i64>  
        extractelement <2 x i64> %4, i32 0 
        bitcast i64 %5 to <4 x i16>        
        store <4 x i16> %6, <4 x i16>* @g2, align 8
        ret void
; X64: 	t10:
; X64: 		pextrw	$4, [[X0:%xmm[0-9]+]], %ecx
; X64: 		pextrw	$6, [[X0]], %eax
; X64: 		movlhps [[X0]], [[X0]]
; X64: 		pshuflw	$8, [[X0]], [[X0]]
; X64: 		pinsrw	$2, %ecx, [[X0]]
; X64: 		pinsrw	$3, %eax, [[X0]]
}


; Pack various elements via shuffles.
define <8 x i16> @t11(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp7 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 1, i32 8, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
	ret <8 x i16> %tmp7

; X64: t11:
; X64:	movd	%xmm1, %eax
; X64:	movlhps	%xmm0, %xmm0
; X64:	pshuflw	$1, %xmm0, %xmm0
; X64:	pinsrw	$1, %eax, %xmm0
; X64:	ret
}


define <8 x i16> @t12(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 0, i32 1, i32 undef, i32 undef, i32 3, i32 11, i32 undef , i32 undef >
	ret <8 x i16> %tmp9

; X64: t12:
; X64: 	pextrw	$3, %xmm1, %eax
; X64: 	movlhps	%xmm0, %xmm0
; X64: 	pshufhw	$3, %xmm0, %xmm0
; X64: 	pinsrw	$5, %eax, %xmm0
; X64: 	ret
}


define <8 x i16> @t13(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 8, i32 9, i32 undef, i32 undef, i32 11, i32 3, i32 undef , i32 undef >
	ret <8 x i16> %tmp9
; X64: t13:
; X64: 	punpcklqdq	%xmm0, %xmm1
; X64: 	pextrw	$3, %xmm1, %eax
; X64: 	pshufd	$52, %xmm1, %xmm0
; X64: 	pinsrw	$4, %eax, %xmm0
; X64: 	ret
}


define <8 x i16> @t14(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
	%tmp9 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 8, i32 9, i32 undef, i32 undef, i32 undef, i32 2, i32 undef , i32 undef >
	ret <8 x i16> %tmp9
; X64: t14:
; X64: 	punpcklqdq	%xmm0, %xmm1
; X64: 	pshufhw	$8, %xmm1, %xmm0
; X64: 	ret
}


; FIXME: t15 is worse off from disabling of scheduler 2-address hack.
define <8 x i16> @t15(<8 x i16> %T0, <8 x i16> %T1) nounwind readnone {
entry:
        %tmp8 = shufflevector <8 x i16> %T0, <8 x i16> %T1, <8 x i32> < i32 undef, i32 undef, i32 7, i32 2, i32 8, i32 undef, i32 undef , i32 undef >
        ret <8 x i16> %tmp8
; X64: 	t15:
; X64: 		pextrw	$7, %xmm0, %eax
; X64: 		punpcklqdq	%xmm1, %xmm0
; X64: 		pshuflw	$-128, %xmm0, %xmm0
; X64: 		pinsrw	$2, %eax, %xmm0
; X64: 		ret
}


; Test yonah where we convert a shuffle to pextrw and pinrsw
define <16 x i8> @t16(<16 x i8> %T0) nounwind readnone {
entry:
        %tmp8 = shufflevector <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 1, i8 1, i8 1, i8 1, i8 0, i8 0, i8 0, i8 0,  i8 0, i8 0, i8 0, i8 0>, <16 x i8> %T0, <16 x i32> < i32 0, i32 1, i32 16, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
        %tmp9 = shufflevector <16 x i8> %tmp8, <16 x i8> %T0,  <16 x i32> < i32 0, i32 1, i32 2, i32 17,  i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef , i32 undef >
        ret <16 x i8> %tmp9
; X64: 	t16:
; X64: 		pextrw	$8, %xmm0, %eax
; X64: 		pslldq	$2, %xmm0
; X64: 		pextrw	$1, %xmm0, %ecx
; X64: 		movzbl	%cl, %ecx
; X64: 		orl	%eax, %ecx
; X64: 		pinsrw	$1, %ecx, %xmm0
; X64: 		ret
}

; rdar://8520311
define <4 x i32> @t17() nounwind {
entry:
; X64: t17:
; X64:          movddup (%rax), %xmm0
  %tmp1 = load <4 x float>* undef, align 16
  %tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  %tmp3 = load <4 x float>* undef, align 16
  %tmp4 = shufflevector <4 x float> %tmp2, <4 x float> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %tmp5 = bitcast <4 x float> %tmp3 to <4 x i32>
  %tmp6 = shufflevector <4 x i32> %tmp5, <4 x i32> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
  %tmp7 = and <4 x i32> %tmp6, <i32 undef, i32 undef, i32 -1, i32 0>
  ret <4 x i32> %tmp7
}
