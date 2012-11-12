; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @test_vrev64D8(<8 x i8>* %A) nounwind {
;CHECK: test_vrev64D8:
;CHECK: vrev64.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>
	ret <8 x i8> %tmp2
}

define <4 x i16> @test_vrev64D16(<4 x i16>* %A) nounwind {
;CHECK: test_vrev64D16:
;CHECK: vrev64.16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
	ret <4 x i16> %tmp2
}

define <2 x i32> @test_vrev64D32(<2 x i32>* %A) nounwind {
;CHECK: test_vrev64D32:
;CHECK: vrev64.32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = shufflevector <2 x i32> %tmp1, <2 x i32> undef, <2 x i32> <i32 1, i32 0>
	ret <2 x i32> %tmp2
}

define <2 x float> @test_vrev64Df(<2 x float>* %A) nounwind {
;CHECK: test_vrev64Df:
;CHECK: vrev64.32
	%tmp1 = load <2 x float>* %A
	%tmp2 = shufflevector <2 x float> %tmp1, <2 x float> undef, <2 x i32> <i32 1, i32 0>
	ret <2 x float> %tmp2
}

define <16 x i8> @test_vrev64Q8(<16 x i8>* %A) nounwind {
;CHECK: test_vrev64Q8:
;CHECK: vrev64.8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0, i32 15, i32 14, i32 13, i32 12, i32 11, i32 10, i32 9, i32 8>
	ret <16 x i8> %tmp2
}

define <8 x i16> @test_vrev64Q16(<8 x i16>* %A) nounwind {
;CHECK: test_vrev64Q16:
;CHECK: vrev64.16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
	ret <8 x i16> %tmp2
}

define <4 x i32> @test_vrev64Q32(<4 x i32>* %A) nounwind {
;CHECK: test_vrev64Q32:
;CHECK: vrev64.32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = shufflevector <4 x i32> %tmp1, <4 x i32> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x i32> %tmp2
}

define <4 x float> @test_vrev64Qf(<4 x float>* %A) nounwind {
;CHECK: test_vrev64Qf:
;CHECK: vrev64.32
	%tmp1 = load <4 x float>* %A
	%tmp2 = shufflevector <4 x float> %tmp1, <4 x float> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x float> %tmp2
}

define <8 x i8> @test_vrev32D8(<8 x i8>* %A) nounwind {
;CHECK: test_vrev32D8:
;CHECK: vrev32.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4>
	ret <8 x i8> %tmp2
}

define <4 x i16> @test_vrev32D16(<4 x i16>* %A) nounwind {
;CHECK: test_vrev32D16:
;CHECK: vrev32.16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = shufflevector <4 x i16> %tmp1, <4 x i16> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
	ret <4 x i16> %tmp2
}

define <16 x i8> @test_vrev32Q8(<16 x i8>* %A) nounwind {
;CHECK: test_vrev32Q8:
;CHECK: vrev32.8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 3, i32 2, i32 1, i32 0, i32 7, i32 6, i32 5, i32 4, i32 11, i32 10, i32 9, i32 8, i32 15, i32 14, i32 13, i32 12>
	ret <16 x i8> %tmp2
}

define <8 x i16> @test_vrev32Q16(<8 x i16>* %A) nounwind {
;CHECK: test_vrev32Q16:
;CHECK: vrev32.16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
	ret <8 x i16> %tmp2
}

define <8 x i8> @test_vrev16D8(<8 x i8>* %A) nounwind {
;CHECK: test_vrev16D8:
;CHECK: vrev16.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6>
	ret <8 x i8> %tmp2
}

define <16 x i8> @test_vrev16Q8(<16 x i8>* %A) nounwind {
;CHECK: test_vrev16Q8:
;CHECK: vrev16.8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = shufflevector <16 x i8> %tmp1, <16 x i8> undef, <16 x i32> <i32 1, i32 0, i32 3, i32 2, i32 5, i32 4, i32 7, i32 6, i32 9, i32 8, i32 11, i32 10, i32 13, i32 12, i32 15, i32 14>
	ret <16 x i8> %tmp2
}

; Undef shuffle indices should not prevent matching to VREV:

define <8 x i8> @test_vrev64D8_undef(<8 x i8>* %A) nounwind {
;CHECK: test_vrev64D8_undef:
;CHECK: vrev64.8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = shufflevector <8 x i8> %tmp1, <8 x i8> undef, <8 x i32> <i32 7, i32 undef, i32 undef, i32 4, i32 3, i32 2, i32 1, i32 0>
	ret <8 x i8> %tmp2
}

define <8 x i16> @test_vrev32Q16_undef(<8 x i16>* %A) nounwind {
;CHECK: test_vrev32Q16_undef:
;CHECK: vrev32.16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 undef, i32 0, i32 undef, i32 2, i32 5, i32 4, i32 7, i32 undef>
	ret <8 x i16> %tmp2
}

; A vcombine feeding a VREV should not obscure things.  Radar 8597007.

define void @test_with_vcombine(<4 x float>* %v) nounwind {
;CHECK: test_with_vcombine:
;CHECK-NOT: vext
;CHECK: vrev64.32
  %tmp1 = load <4 x float>* %v, align 16
  %tmp2 = bitcast <4 x float> %tmp1 to <2 x double>
  %tmp3 = extractelement <2 x double> %tmp2, i32 0
  %tmp4 = bitcast double %tmp3 to <2 x float>
  %tmp5 = extractelement <2 x double> %tmp2, i32 1
  %tmp6 = bitcast double %tmp5 to <2 x float>
  %tmp7 = fadd <2 x float> %tmp6, %tmp6
  %tmp8 = shufflevector <2 x float> %tmp4, <2 x float> %tmp7, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  store <4 x float> %tmp8, <4 x float>* %v, align 16
  ret void
}

; The type <2 x i16> is legalized to <2 x i32> and need to be trunc-stored
; to <2 x i16> when stored to memory.
define void @test_vrev64(<4 x i16>* nocapture %source, <2 x i16>* nocapture %dst) nounwind ssp {
; CHECK: test_vrev64:
; CHECK: vst1.32
entry:
  %0 = bitcast <4 x i16>* %source to <8 x i16>*
  %tmp2 = load <8 x i16>* %0, align 4
  %tmp3 = extractelement <8 x i16> %tmp2, i32 6
  %tmp5 = insertelement <2 x i16> undef, i16 %tmp3, i32 0
  %tmp9 = extractelement <8 x i16> %tmp2, i32 5
  %tmp11 = insertelement <2 x i16> %tmp5, i16 %tmp9, i32 1
  store <2 x i16> %tmp11, <2 x i16>* %dst, align 4
  ret void
}

; Test vrev of float4
define void @float_vrev64(float* nocapture %source, <4 x float>* nocapture %dest) nounwind noinline ssp {
; CHECK: float_vrev64
; CHECK: vext.32
; CHECK: vrev64.32
entry:
  %0 = bitcast float* %source to <4 x float>*
  %tmp2 = load <4 x float>* %0, align 4
  %tmp5 = shufflevector <4 x float> <float 0.000000e+00, float undef, float undef, float undef>, <4 x float> %tmp2, <4 x i32> <i32 0, i32 7, i32 0, i32 0>
  %arrayidx8 = getelementptr inbounds <4 x float>* %dest, i32 11
  store <4 x float> %tmp5, <4 x float>* %arrayidx8, align 4
  ret void
}

