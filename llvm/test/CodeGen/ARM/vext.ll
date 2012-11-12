; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

define <8 x i8> @test_vextd(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: test_vextd:
;CHECK: vext
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
	ret <8 x i8> %tmp3
}

define <8 x i8> @test_vextRd(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: test_vextRd:
;CHECK: vext
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4>
	ret <8 x i8> %tmp3
}

define <16 x i8> @test_vextq(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: test_vextq:
;CHECK: vext
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
	ret <16 x i8> %tmp3
}

define <16 x i8> @test_vextRq(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: test_vextRq:
;CHECK: vext
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6>
	ret <16 x i8> %tmp3
}

define <4 x i16> @test_vextd16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: test_vextd16:
;CHECK: vext
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
	ret <4 x i16> %tmp3
}

define <4 x i32> @test_vextq32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: test_vextq32:
;CHECK: vext
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
	ret <4 x i32> %tmp3
}

; Undef shuffle indices should not prevent matching to VEXT:

define <8 x i8> @test_vextd_undef(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: test_vextd_undef:
;CHECK: vext
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 3, i32 undef, i32 undef, i32 6, i32 7, i32 8, i32 9, i32 10>
	ret <8 x i8> %tmp3
}

define <16 x i8> @test_vextRq_undef(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: test_vextRq_undef:
;CHECK: vext
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 23, i32 24, i32 25, i32 26, i32 undef, i32 undef, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 undef, i32 6>
	ret <16 x i8> %tmp3
}

define <16 x i8> @test_vextq_undef_op2(<16 x i8> %a) nounwind {
;CHECK: test_vextq_undef_op2:
;CHECK: vext
entry:
  %tmp1 = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 0, i32 1>
  ret <16 x i8> %tmp1
}

define <8 x i8> @test_vextd_undef_op2(<8 x i8> %a) nounwind {
;CHECK: test_vextd_undef_op2:
;CHECK: vext
entry:
  %tmp1 = shufflevector <8 x i8> %a, <8 x i8> undef, <8 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1>
  ret <8 x i8> %tmp1
}


define <16 x i8> @test_vextq_undef_op2_undef(<16 x i8> %a) nounwind {
;CHECK: test_vextq_undef_op2_undef:
;CHECK: vext
entry:
  %tmp1 = shufflevector <16 x i8> %a, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 4, i32 undef, i32 undef, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 0, i32 1>
  ret <16 x i8> %tmp1
}

define <8 x i8> @test_vextd_undef_op2_undef(<8 x i8> %a) nounwind {
;CHECK: test_vextd_undef_op2_undef:
;CHECK: vext
entry:
  %tmp1 = shufflevector <8 x i8> %a, <8 x i8> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 6, i32 7, i32 undef, i32 1>
  ret <8 x i8> %tmp1
}

; Tests for ReconstructShuffle function. Indices have to be carefully
; chosen to reach lowering phase as a BUILD_VECTOR.

; One vector needs vext, the other can be handled by extract_subvector
; Also checks interleaving of sources is handled correctly.
; Essence: a vext is used on %A and something saner than stack load/store for final result.
define <4 x i16> @test_interleaved(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: test_interleaved:
;CHECK: vext.16
;CHECK-NOT: vext.16
;CHECK: vzip.16
        %tmp1 = load <8 x i16>* %A
        %tmp2 = load <8 x i16>* %B
        %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <4 x i32> <i32 3, i32 8, i32 5, i32 9>
        ret <4 x i16> %tmp3
}

; An undef in the shuffle list should still be optimizable
define <4 x i16> @test_undef(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: test_undef:
;CHECK: vzip.16
        %tmp1 = load <8 x i16>* %A
        %tmp2 = load <8 x i16>* %B
        %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <4 x i32> <i32 undef, i32 8, i32 5, i32 9>
        ret <4 x i16> %tmp3
}

; We should ignore a build_vector with more than two sources.
; Use illegal <32 x i16> type to produce such a shuffle after legalizing types.
; Try to look for fallback to stack expansion.
define <4 x i16> @test_multisource(<32 x i16>* %B) nounwind {
;CHECK: test_multisource:
;CHECK: vst1.16
        %tmp1 = load <32 x i16>* %B
        %tmp2 = shufflevector <32 x i16> %tmp1, <32 x i16> undef, <4 x i32> <i32 0, i32 8, i32 16, i32 24>
        ret <4 x i16> %tmp2
}

; We don't handle shuffles using more than half of a 128-bit vector.
; Again, test for fallback to stack expansion
define <4 x i16> @test_largespan(<8 x i16>* %B) nounwind {
;CHECK: test_largespan:
;CHECK: vst1.16
        %tmp1 = load <8 x i16>* %B
        %tmp2 = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
        ret <4 x i16> %tmp2
}

; The actual shuffle code only handles some cases, make sure we check
; this rather than blindly emitting a VECTOR_SHUFFLE (infinite
; lowering loop can result otherwise).
define <8 x i16> @test_illegal(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: test_illegal:
;CHECK: vst1.16
       %tmp1 = load <8 x i16>* %A
       %tmp2 = load <8 x i16>* %B
       %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <8 x i32> <i32 0, i32 7, i32 5, i32 13, i32 3, i32 2, i32 2, i32 9>
       ret <8 x i16> %tmp3
}

; PR11129
; Make sure this doesn't crash
define arm_aapcscc void @test_elem_mismatch(<2 x i64>* nocapture %src, <4 x i16>* nocapture %dest) nounwind {
; CHECK: test_elem_mismatch:
; CHECK: vstr
  %tmp0 = load <2 x i64>* %src, align 16
  %tmp1 = bitcast <2 x i64> %tmp0 to <4 x i32>
  %tmp2 = extractelement <4 x i32> %tmp1, i32 0
  %tmp3 = extractelement <4 x i32> %tmp1, i32 2
  %tmp4 = trunc i32 %tmp2 to i16
  %tmp5 = trunc i32 %tmp3 to i16
  %tmp6 = insertelement <4 x i16> undef, i16 %tmp4, i32 0
  %tmp7 = insertelement <4 x i16> %tmp6, i16 %tmp5, i32 1
  store <4 x i16> %tmp7, <4 x i16>* %dest, align 4
  ret void
}
