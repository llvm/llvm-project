; RUN: llc -mattr=+neon < %s | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:64:64-v128:128:128-a0:0:32"
target triple = "thumbv7-elf"

define <4 x i16> @vqdmulhs16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vqdmulhs16:
;CHECK: vqdmulh.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqdmulhs32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vqdmulhs32:
;CHECK: vqdmulh.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i16> @vqdmulhQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vqdmulhQs16:
;CHECK: vqdmulh.s16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqdmulhQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vqdmulhQs32:
;CHECK: vqdmulh.s32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define arm_aapcs_vfpcc <8 x i16> @test_vqdmulhQ_lanes16(<8 x i16> %arg0_int16x8_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vqdmulhQ_lanes16
; CHECK: vqdmulh.s16 q0, q0, d2[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1> ; <<8 x i16>> [#uses=1]
  %1 = tail call <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16> %arg0_int16x8_t, <8 x i16> %0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %1
}

define arm_aapcs_vfpcc <4 x i32> @test_vqdmulhQ_lanes32(<4 x i32> %arg0_int32x4_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vqdmulhQ_lanes32
; CHECK: vqdmulh.s32 q0, q0, d2[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i32>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32> %arg0_int32x4_t, <4 x i32> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <4 x i16> @test_vqdmulh_lanes16(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vqdmulh_lanes16
; CHECK: vqdmulh.s16 d0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16> %arg0_int16x4_t, <4 x i16> %0) ; <<4 x i16>> [#uses=1]
  ret <4 x i16> %1
}

define arm_aapcs_vfpcc <2 x i32> @test_vqdmulh_lanes32(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vqdmulh_lanes32
; CHECK: vqdmulh.s32 d0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32> %arg0_int32x2_t, <2 x i32> %0) ; <<2 x i32>> [#uses=1]
  ret <2 x i32> %1
}

declare <4 x i16> @llvm.arm.neon.vqdmulh.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqdmulh.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vqdmulh.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqdmulh.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i16> @vqrdmulhs16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vqrdmulhs16:
;CHECK: vqrdmulh.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i16> %tmp3
}

define <2 x i32> @vqrdmulhs32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vqrdmulhs32:
;CHECK: vqrdmulh.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i32> %tmp3
}

define <8 x i16> @vqrdmulhQs16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK-LABEL: vqrdmulhQs16:
;CHECK: vqrdmulh.s16
	%tmp1 = load <8 x i16>, <8 x i16>* %A
	%tmp2 = load <8 x i16>, <8 x i16>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %tmp1, <8 x i16> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vqrdmulhQs32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK-LABEL: vqrdmulhQs32:
;CHECK: vqrdmulh.s32
	%tmp1 = load <4 x i32>, <4 x i32>* %A
	%tmp2 = load <4 x i32>, <4 x i32>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp2)
	ret <4 x i32> %tmp3
}

define arm_aapcs_vfpcc <8 x i16> @test_vqRdmulhQ_lanes16(<8 x i16> %arg0_int16x8_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vqRdmulhQ_lanes16
; CHECK: vqrdmulh.s16 q0, q0, d2[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1> ; <<8 x i16>> [#uses=1]
  %1 = tail call <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16> %arg0_int16x8_t, <8 x i16> %0) ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %1
}

define arm_aapcs_vfpcc <4 x i32> @test_vqRdmulhQ_lanes32(<4 x i32> %arg0_int32x4_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vqRdmulhQ_lanes32
; CHECK: vqrdmulh.s32 q0, q0, d2[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i32>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32> %arg0_int32x4_t, <4 x i32> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <4 x i16> @test_vqRdmulh_lanes16(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vqRdmulh_lanes16
; CHECK: vqrdmulh.s16 d0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16> %arg0_int16x4_t, <4 x i16> %0) ; <<4 x i16>> [#uses=1]
  ret <4 x i16> %1
}

define arm_aapcs_vfpcc <2 x i32> @test_vqRdmulh_lanes32(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vqRdmulh_lanes32
; CHECK: vqrdmulh.s32 d0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32> %arg0_int32x2_t, <2 x i32> %0) ; <<2 x i32>> [#uses=1]
  ret <2 x i32> %1
}

declare <4 x i16> @llvm.arm.neon.vqrdmulh.v4i16(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i32> @llvm.arm.neon.vqrdmulh.v2i32(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vqrdmulh.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vqrdmulh.v4i32(<4 x i32>, <4 x i32>) nounwind readnone

define <4 x i32> @vqdmulls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK-LABEL: vqdmulls16:
;CHECK: vqdmull.s16
	%tmp1 = load <4 x i16>, <4 x i16>* %A
	%tmp2 = load <4 x i16>, <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vqdmulls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK-LABEL: vqdmulls32:
;CHECK: vqdmull.s32
	%tmp1 = load <2 x i32>, <2 x i32>* %A
	%tmp2 = load <2 x i32>, <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define arm_aapcs_vfpcc <4 x i32> @test_vqdmull_lanes16(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vqdmull_lanes16
; CHECK: vqdmull.s16 q0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %arg0_int16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vqdmull_lanes32(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vqdmull_lanes32
; CHECK: vqdmull.s32 q0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %arg0_int32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

declare <4 x i32>  @llvm.arm.neon.vqdmull.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64>  @llvm.arm.neon.vqdmull.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

define <4 x i32> @vqdmlals16_natural(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK-LABEL: vqdmlals16_natural:
;CHECK: vqdmlal.s16
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = load <4 x i16>, <4 x i16>* %C
        %tmp4 = call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %tmp2, <4 x i16> %tmp3)
        %tmp5 = call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp4)
        ret <4 x i32> %tmp5
}

define <2 x i64> @vqdmlals32_natural(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK-LABEL: vqdmlals32_natural:
;CHECK: vqdmlal.s32
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = load <2 x i32>, <2 x i32>* %C
        %tmp4 = call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %tmp2, <2 x i32> %tmp3)
        %tmp5 = call <2 x i64> @llvm.sadd.sat.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp4)
        ret <2 x i64> %tmp5
}

define arm_aapcs_vfpcc <4 x i32> @test_vqdmlal_lanes16_natural(<4 x i32> %arg0_int32x4_t, <4 x i16> %arg1_int16x4_t, <4 x i16> %arg2_int16x4_t) nounwind readnone {
entry:
; CHECK-LABEL: test_vqdmlal_lanes16_natural:
; CHECK: vqdmlal.s16 q0, d2, d3[1]
  %0 = shufflevector <4 x i16> %arg2_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %arg1_int16x4_t, <4 x i16> %0)
  %2 = tail call <4 x i32> @llvm.sadd.sat.v4i32(<4 x i32> %arg0_int32x4_t, <4 x i32> %1)
  ret <4 x i32> %2
}

define arm_aapcs_vfpcc <2 x i64> @test_vqdmlal_lanes32_natural(<2 x i64> %arg0_int64x2_t, <2 x i32> %arg1_int32x2_t, <2 x i32> %arg2_int32x2_t) nounwind readnone {
entry:
; CHECK-LABEL: test_vqdmlal_lanes32_natural:
; CHECK: vqdmlal.s32 q0, d2, d3[1]
  %0 = shufflevector <2 x i32> %arg2_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %arg1_int32x2_t, <2 x i32> %0)
  %2 = call <2 x i64> @llvm.sadd.sat.v2i64(<2 x i64> %arg0_int64x2_t, <2 x i64> %1)
  ret <2 x i64> %2
}

declare <4 x i32>  @llvm.sadd.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64>  @llvm.sadd.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone

define <4 x i32> @vqdmlsls16_natural(<4 x i32>* %A, <4 x i16>* %B, <4 x i16>* %C) nounwind {
;CHECK-LABEL: vqdmlsls16_natural:
;CHECK: vqdmlsl.s16
        %tmp1 = load <4 x i32>, <4 x i32>* %A
        %tmp2 = load <4 x i16>, <4 x i16>* %B
        %tmp3 = load <4 x i16>, <4 x i16>* %C
        %tmp4 = call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %tmp2, <4 x i16> %tmp3)
        %tmp5 = call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %tmp1, <4 x i32> %tmp4)
        ret <4 x i32> %tmp5
}

define <2 x i64> @vqdmlsls32_natural(<2 x i64>* %A, <2 x i32>* %B, <2 x i32>* %C) nounwind {
;CHECK-LABEL: vqdmlsls32_natural:
;CHECK: vqdmlsl.s32
        %tmp1 = load <2 x i64>, <2 x i64>* %A
        %tmp2 = load <2 x i32>, <2 x i32>* %B
        %tmp3 = load <2 x i32>, <2 x i32>* %C
        %tmp4 = call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %tmp2, <2 x i32> %tmp3)
        %tmp5 = call <2 x i64> @llvm.ssub.sat.v2i64(<2 x i64> %tmp1, <2 x i64> %tmp4)
        ret <2 x i64> %tmp5
}

define arm_aapcs_vfpcc <4 x i32> @test_vqdmlsl_lanes16_natural(<4 x i32> %arg0_int32x4_t, <4 x i16> %arg1_int16x4_t, <4 x i16> %arg2_int16x4_t) nounwind readnone {
entry:
; CHECK-LABEL: test_vqdmlsl_lanes16_natural:
; CHECK: vqdmlsl.s16 q0, d2, d3[1]
  %0 = shufflevector <4 x i16> %arg2_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vqdmull.v4i32(<4 x i16> %arg1_int16x4_t, <4 x i16> %0)
  %2 = tail call <4 x i32> @llvm.ssub.sat.v4i32(<4 x i32> %arg0_int32x4_t, <4 x i32> %1)
  ret <4 x i32> %2
}

define arm_aapcs_vfpcc <2 x i64> @test_vqdmlsl_lanes32_natural(<2 x i64> %arg0_int64x2_t, <2 x i32> %arg1_int32x2_t, <2 x i32> %arg2_int32x2_t) nounwind readnone {
entry:
; CHECK-LABEL: test_vqdmlsl_lanes32_natural:
; CHECK: vqdmlsl.s32 q0, d2, d3[1]
  %0 = shufflevector <2 x i32> %arg2_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vqdmull.v2i64(<2 x i32> %arg1_int32x2_t, <2 x i32> %0)
  %2 = call <2 x i64> @llvm.ssub.sat.v2i64(<2 x i64> %arg0_int64x2_t, <2 x i64> %1)
  ret <2 x i64> %2
}

declare <4 x i32>  @llvm.ssub.sat.v4i32(<4 x i32>, <4 x i32>) nounwind readnone
declare <2 x i64>  @llvm.ssub.sat.v2i64(<2 x i64>, <2 x i64>) nounwind readnone
