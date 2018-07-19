; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s

define <8 x i8> @test_vextd(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: test_vextd:
;CHECK: {{ext.8b.*#3}}
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10>
	ret <8 x i8> %tmp3
}

define <8 x i8> @test_vextRd(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: test_vextRd:
;CHECK: {{ext.8b.*#5}}
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 13, i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4>
	ret <8 x i8> %tmp3
}

define <16 x i8> @test_vextq(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: test_vextq:
;CHECK: {{ext.16b.*3}}
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18>
	ret <16 x i8> %tmp3
}

define <16 x i8> @test_vextRq(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: test_vextRq:
;CHECK: {{ext.16b.*7}}
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6>
	ret <16 x i8> %tmp3
}

define <4 x i16> @test_vextd16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: test_vextd16:
;CHECK: {{ext.8b.*#6}}
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = shufflevector <4 x i16> %tmp1, <4 x i16> %tmp2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
	ret <4 x i16> %tmp3
}

define <4 x i32> @test_vextq32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: test_vextq32:
;CHECK: {{ext.16b.*12}}
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = shufflevector <4 x i32> %tmp1, <4 x i32> %tmp2, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
	ret <4 x i32> %tmp3
}

; Undef shuffle indices should not prevent matching to VEXT:

define <8 x i8> @test_vextd_undef(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: test_vextd_undef:
;CHECK: {{ext.8b.*}}
	%tmp1 = load <8 x i8>, <8 x i8>* %A
	%tmp2 = load <8 x i8>, <8 x i8>* %B
	%tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 3, i32 undef, i32 undef, i32 6, i32 7, i32 8, i32 9, i32 10>
	ret <8 x i8> %tmp3
}

define <8 x i8> @test_vextd_undef2(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK-LABEL: test_vextd_undef2:
;CHECK: {{ext.8b.*#6}}
  %tmp1 = load <8 x i8>, <8 x i8>* %A
  %tmp2 = load <8 x i8>, <8 x i8>* %B
  %tmp3 = shufflevector <8 x i8> %tmp1, <8 x i8> %tmp2, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 2, i32 3, i32 4, i32 5>
  ret <8 x i8> %tmp3
}

define <16 x i8> @test_vextRq_undef(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK-LABEL: test_vextRq_undef:
;CHECK: {{ext.16b.*#7}}
	%tmp1 = load <16 x i8>, <16 x i8>* %A
	%tmp2 = load <16 x i8>, <16 x i8>* %B
	%tmp3 = shufflevector <16 x i8> %tmp1, <16 x i8> %tmp2, <16 x i32> <i32 23, i32 24, i32 25, i32 26, i32 undef, i32 undef, i32 29, i32 30, i32 31, i32 0, i32 1, i32 2, i32 3, i32 4, i32 undef, i32 6>
	ret <16 x i8> %tmp3
}

define <8 x i16> @test_vextRq_undef2(<8 x i16>* %A) nounwind {
;CHECK-LABEL: test_vextRq_undef2:
;CHECK: {{ext.16b.*#10}}
  %tmp1 = load <8 x i16>, <8 x i16>* %A
  %vext = shufflevector <8 x i16> %tmp1, <8 x i16> undef, <8 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 1, i32 2, i32 3, i32 4>
  ret <8 x i16> %vext;
}

; Tests for ReconstructShuffle function. Indices have to be carefully
; chosen to reach lowering phase as a BUILD_VECTOR.

; One vector needs vext, the other can be handled by extract_subvector
; Also checks interleaving of sources is handled correctly.
; Essence: a vext is used on %A and something saner than stack load/store for final result.
define <4 x i16> @test_interleaved(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: test_interleaved:
;CHECK: ext.8b
;CHECK: zip1.4h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <4 x i32> <i32 3, i32 8, i32 5, i32 9>
        ret <4 x i16> %tmp3
}

; An undef in the shuffle list should still be optimizable
define <4 x i16> @test_undef(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: test_undef:
;CHECK: zip1.4h
        %tmp1 = load <8 x i16>, <8 x i16>* %A
        %tmp2 = load <8 x i16>, <8 x i16>* %B
        %tmp3 = shufflevector <8 x i16> %tmp1, <8 x i16> %tmp2, <4 x i32> <i32 undef, i32 8, i32 5, i32 9>
        ret <4 x i16> %tmp3
}
