; RUN: llc < %s -march=arm -mattr=+neon | FileCheck %s

; rdar://12471808

define <8 x i8> @v_bsli8(<8 x i8>* %A, <8 x i8>* %B, <8 x i8>* %C) nounwind {
;CHECK: v_bsli8:
;CHECK: vbsl
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = load <8 x i8>* %C
	%tmp4 = and <8 x i8> %tmp1, %tmp2
	%tmp5 = xor <8 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp6 = and <8 x i8> %tmp5, %tmp3
	%tmp7 = or <8 x i8> %tmp4, %tmp6
	ret <8 x i8> %tmp7
}

define <4 x i16> @v_bsli16(<4 x i16>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK: v_bsli16:
;CHECK: vbsl
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = load <4 x i16>* %C
	%tmp4 = and <4 x i16> %tmp1, %tmp2
	%tmp5 = xor <4 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp6 = and <4 x i16> %tmp5, %tmp3
	%tmp7 = or <4 x i16> %tmp4, %tmp6
	ret <4 x i16> %tmp7
}

define <2 x i32> @v_bsli32(<2 x i32>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK: v_bsli32:
;CHECK: vbsl
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = load <2 x i32>* %C
	%tmp4 = and <2 x i32> %tmp1, %tmp2
	%tmp5 = xor <2 x i32> %tmp1, < i32 -1, i32 -1 >
	%tmp6 = and <2 x i32> %tmp5, %tmp3
	%tmp7 = or <2 x i32> %tmp4, %tmp6
	ret <2 x i32> %tmp7
}

define <1 x i64> @v_bsli64(<1 x i64>* %A, <1 x i64>* %B, <1 x i64>* %C) nounwind {
;CHECK: v_bsli64:
;CHECK: vbsl
	%tmp1 = load <1 x i64>* %A
	%tmp2 = load <1 x i64>* %B
	%tmp3 = load <1 x i64>* %C
	%tmp4 = and <1 x i64> %tmp1, %tmp2
	%tmp5 = xor <1 x i64> %tmp1, < i64 -1 >
	%tmp6 = and <1 x i64> %tmp5, %tmp3
	%tmp7 = or <1 x i64> %tmp4, %tmp6
	ret <1 x i64> %tmp7
}

define <16 x i8> @v_bslQi8(<16 x i8>* %A, <16 x i8>* %B, <16 x i8>* %C) nounwind {
;CHECK: v_bslQi8:
;CHECK: vbsl
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = load <16 x i8>* %C
	%tmp4 = and <16 x i8> %tmp1, %tmp2
	%tmp5 = xor <16 x i8> %tmp1, < i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1, i8 -1 >
	%tmp6 = and <16 x i8> %tmp5, %tmp3
	%tmp7 = or <16 x i8> %tmp4, %tmp6
	ret <16 x i8> %tmp7
}

define <8 x i16> @v_bslQi16(<8 x i16>* %A, <8 x i16>* %B, <8 x i16>* %C) nounwind {
;CHECK: v_bslQi16:
;CHECK: vbsl
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = load <8 x i16>* %C
	%tmp4 = and <8 x i16> %tmp1, %tmp2
	%tmp5 = xor <8 x i16> %tmp1, < i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1, i16 -1 >
	%tmp6 = and <8 x i16> %tmp5, %tmp3
	%tmp7 = or <8 x i16> %tmp4, %tmp6
	ret <8 x i16> %tmp7
}

define <4 x i32> @v_bslQi32(<4 x i32>* %A, <4 x i32>* %B, <4 x i32>* %C) nounwind {
;CHECK: v_bslQi32:
;CHECK: vbsl
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = load <4 x i32>* %C
	%tmp4 = and <4 x i32> %tmp1, %tmp2
	%tmp5 = xor <4 x i32> %tmp1, < i32 -1, i32 -1, i32 -1, i32 -1 >
	%tmp6 = and <4 x i32> %tmp5, %tmp3
	%tmp7 = or <4 x i32> %tmp4, %tmp6
	ret <4 x i32> %tmp7
}

define <2 x i64> @v_bslQi64(<2 x i64>* %A, <2 x i64>* %B, <2 x i64>* %C) nounwind {
;CHECK: v_bslQi64:
;CHECK: vbsl
	%tmp1 = load <2 x i64>* %A
	%tmp2 = load <2 x i64>* %B
	%tmp3 = load <2 x i64>* %C
	%tmp4 = and <2 x i64> %tmp1, %tmp2
	%tmp5 = xor <2 x i64> %tmp1, < i64 -1, i64 -1 >
	%tmp6 = and <2 x i64> %tmp5, %tmp3
	%tmp7 = or <2 x i64> %tmp4, %tmp6
	ret <2 x i64> %tmp7
}

define <8 x i8> @f1(<8 x i8> %a, <8 x i8> %b, <8 x i8> %c) nounwind readnone optsize ssp {
; CHECK: f1:
; CHECK: vbsl
  %vbsl.i = tail call <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8> %a, <8 x i8> %b, <8 x i8> %c) nounwind
  ret <8 x i8> %vbsl.i
}

define <4 x i16> @f2(<4 x i16> %a, <4 x i16> %b, <4 x i16> %c) nounwind readnone optsize ssp {
; CHECK: f2:
; CHECK: vbsl
  %vbsl3.i = tail call <4 x i16> @llvm.arm.neon.vbsl.v4i16(<4 x i16> %a, <4 x i16> %b, <4 x i16> %c) nounwind
  ret <4 x i16> %vbsl3.i
}

define <2 x i32> @f3(<2 x i32> %a, <2 x i32> %b, <2 x i32> %c) nounwind readnone optsize ssp {
; CHECK: f3:
; CHECK: vbsl
  %vbsl3.i = tail call <2 x i32> @llvm.arm.neon.vbsl.v2i32(<2 x i32> %a, <2 x i32> %b, <2 x i32> %c) nounwind
  ret <2 x i32> %vbsl3.i
}

define <2 x float> @f4(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind readnone optsize ssp {
; CHECK: f4:
; CHECK: vbsl
  %vbsl4.i = tail call <2 x float> @llvm.arm.neon.vbsl.v2f32(<2 x float> %a, <2 x float> %b, <2 x float> %c) nounwind
  ret <2 x float> %vbsl4.i
}

define <16 x i8> @g1(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) nounwind readnone optsize ssp {
; CHECK: g1:
; CHECK: vbsl
  %vbsl.i = tail call <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) nounwind
  ret <16 x i8> %vbsl.i
}

define <8 x i16> @g2(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) nounwind readnone optsize ssp {
; CHECK: g2:
; CHECK: vbsl
  %vbsl3.i = tail call <8 x i16> @llvm.arm.neon.vbsl.v8i16(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) nounwind
  ret <8 x i16> %vbsl3.i
}

define <4 x i32> @g3(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) nounwind readnone optsize ssp {
; CHECK: g3:
; CHECK: vbsl
  %vbsl3.i = tail call <4 x i32> @llvm.arm.neon.vbsl.v4i32(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) nounwind
  ret <4 x i32> %vbsl3.i
}

define <4 x float> @g4(<4 x float> %a, <4 x float> %b, <4 x float> %c) nounwind readnone optsize ssp {
; CHECK: g4:
; CHECK: vbsl
  %vbsl4.i = tail call <4 x float> @llvm.arm.neon.vbsl.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c) nounwind
  ret <4 x float> %vbsl4.i
}

define <1 x i64> @test_vbsl_s64(<1 x i64> %a, <1 x i64> %b, <1 x i64> %c) nounwind readnone optsize ssp {
; CHECK: test_vbsl_s64:
; CHECK: vbsl d
  %vbsl3.i = tail call <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64> %a, <1 x i64> %b, <1 x i64> %c) nounwind
  ret <1 x i64> %vbsl3.i
}

define <1 x i64> @test_vbsl_u64(<1 x i64> %a, <1 x i64> %b, <1 x i64> %c) nounwind readnone optsize ssp {
; CHECK: test_vbsl_u64:
; CHECK: vbsl d
  %vbsl3.i = tail call <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64> %a, <1 x i64> %b, <1 x i64> %c) nounwind
  ret <1 x i64> %vbsl3.i
}

define <2 x i64> @test_vbslq_s64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) nounwind readnone optsize ssp {
; CHECK: test_vbslq_s64:
; CHECK: vbsl q
  %vbsl3.i = tail call <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) nounwind
  ret <2 x i64> %vbsl3.i
}

define <2 x i64> @test_vbslq_u64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) nounwind readnone optsize ssp {
; CHECK: test_vbslq_u64:
; CHECK: vbsl q
  %vbsl3.i = tail call <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) nounwind
  ret <2 x i64> %vbsl3.i
}

declare <4 x i32> @llvm.arm.neon.vbsl.v4i32(<4 x i32>, <4 x i32>, <4 x i32>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vbsl.v8i16(<8 x i16>, <8 x i16>, <8 x i16>) nounwind readnone
declare <16 x i8> @llvm.arm.neon.vbsl.v16i8(<16 x i8>, <16 x i8>, <16 x i8>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vbsl.v2i32(<2 x i32>, <2 x i32>, <2 x i32>) nounwind readnone
declare <4 x i16> @llvm.arm.neon.vbsl.v4i16(<4 x i16>, <4 x i16>, <4 x i16>) nounwind readnone
declare <8 x i8> @llvm.arm.neon.vbsl.v8i8(<8 x i8>, <8 x i8>, <8 x i8>) nounwind readnone
declare <2 x float> @llvm.arm.neon.vbsl.v2f32(<2 x float>, <2 x float>, <2 x float>) nounwind readnone
declare <4 x float> @llvm.arm.neon.vbsl.v4f32(<4 x float>, <4 x float>, <4 x float>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vbsl.v2i64(<2 x i64>, <2 x i64>, <2 x i64>) nounwind readnone
declare <1 x i64> @llvm.arm.neon.vbsl.v1i64(<1 x i64>, <1 x i64>, <1 x i64>) nounwind readnone
