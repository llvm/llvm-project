; RUN: llc < %s -march=arm -mcpu=cortex-a8 | FileCheck %s

define <8 x i8> @vmuli8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmuli8:
;CHECK: vmul.i8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = mul <8 x i8> %tmp1, %tmp2
	ret <8 x i8> %tmp3
}

define <4 x i16> @vmuli16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmuli16:
;CHECK: vmul.i16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = mul <4 x i16> %tmp1, %tmp2
	ret <4 x i16> %tmp3
}

define <2 x i32> @vmuli32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmuli32:
;CHECK: vmul.i32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = mul <2 x i32> %tmp1, %tmp2
	ret <2 x i32> %tmp3
}

define <2 x float> @vmulf32(<2 x float>* %A, <2 x float>* %B) nounwind {
;CHECK: vmulf32:
;CHECK: vmul.f32
	%tmp1 = load <2 x float>* %A
	%tmp2 = load <2 x float>* %B
	%tmp3 = fmul <2 x float> %tmp1, %tmp2
	ret <2 x float> %tmp3
}

define <8 x i8> @vmulp8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmulp8:
;CHECK: vmul.p8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i8> @llvm.arm.neon.vmulp.v8i8(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i8> %tmp3
}

define <16 x i8> @vmulQi8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vmulQi8:
;CHECK: vmul.i8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = mul <16 x i8> %tmp1, %tmp2
	ret <16 x i8> %tmp3
}

define <8 x i16> @vmulQi16(<8 x i16>* %A, <8 x i16>* %B) nounwind {
;CHECK: vmulQi16:
;CHECK: vmul.i16
	%tmp1 = load <8 x i16>* %A
	%tmp2 = load <8 x i16>* %B
	%tmp3 = mul <8 x i16> %tmp1, %tmp2
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmulQi32(<4 x i32>* %A, <4 x i32>* %B) nounwind {
;CHECK: vmulQi32:
;CHECK: vmul.i32
	%tmp1 = load <4 x i32>* %A
	%tmp2 = load <4 x i32>* %B
	%tmp3 = mul <4 x i32> %tmp1, %tmp2
	ret <4 x i32> %tmp3
}

define <4 x float> @vmulQf32(<4 x float>* %A, <4 x float>* %B) nounwind {
;CHECK: vmulQf32:
;CHECK: vmul.f32
	%tmp1 = load <4 x float>* %A
	%tmp2 = load <4 x float>* %B
	%tmp3 = fmul <4 x float> %tmp1, %tmp2
	ret <4 x float> %tmp3
}

define <16 x i8> @vmulQp8(<16 x i8>* %A, <16 x i8>* %B) nounwind {
;CHECK: vmulQp8:
;CHECK: vmul.p8
	%tmp1 = load <16 x i8>* %A
	%tmp2 = load <16 x i8>* %B
	%tmp3 = call <16 x i8> @llvm.arm.neon.vmulp.v16i8(<16 x i8> %tmp1, <16 x i8> %tmp2)
	ret <16 x i8> %tmp3
}

declare <8 x i8>  @llvm.arm.neon.vmulp.v8i8(<8 x i8>, <8 x i8>) nounwind readnone
declare <16 x i8>  @llvm.arm.neon.vmulp.v16i8(<16 x i8>, <16 x i8>) nounwind readnone

define arm_aapcs_vfpcc <2 x float> @test_vmul_lanef32(<2 x float> %arg0_float32x2_t, <2 x float> %arg1_float32x2_t) nounwind readnone {
entry:
; CHECK: test_vmul_lanef32:
; CHECK: vmul.f32 d0, d0, d1[0]
  %0 = shufflevector <2 x float> %arg1_float32x2_t, <2 x float> undef, <2 x i32> zeroinitializer ; <<2 x float>> [#uses=1]
  %1 = fmul <2 x float> %0, %arg0_float32x2_t     ; <<2 x float>> [#uses=1]
  ret <2 x float> %1
}

define arm_aapcs_vfpcc <4 x i16> @test_vmul_lanes16(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmul_lanes16:
; CHECK: vmul.i16 d0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses$
  %1 = mul <4 x i16> %0, %arg0_int16x4_t          ; <<4 x i16>> [#uses=1]
  ret <4 x i16> %1
}

define arm_aapcs_vfpcc <2 x i32> @test_vmul_lanes32(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmul_lanes32:
; CHECK: vmul.i32 d0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = mul <2 x i32> %0, %arg0_int32x2_t          ; <<2 x i32>> [#uses=1]
  ret <2 x i32> %1
}

define arm_aapcs_vfpcc <4 x float> @test_vmulQ_lanef32(<4 x float> %arg0_float32x4_t, <2 x float> %arg1_float32x2_t) nounwind readnone {
entry:
; CHECK: test_vmulQ_lanef32:
; CHECK: vmul.f32 q0, q0, d2[1]
  %0 = shufflevector <2 x float> %arg1_float32x2_t, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x float>$
  %1 = fmul <4 x float> %0, %arg0_float32x4_t     ; <<4 x float>> [#uses=1]
  ret <4 x float> %1
}

define arm_aapcs_vfpcc <8 x i16> @test_vmulQ_lanes16(<8 x i16> %arg0_int16x8_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmulQ_lanes16:
; CHECK: vmul.i16 q0, q0, d2[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>
  %1 = mul <8 x i16> %0, %arg0_int16x8_t          ; <<8 x i16>> [#uses=1]
  ret <8 x i16> %1
}

define arm_aapcs_vfpcc <4 x i32> @test_vmulQ_lanes32(<4 x i32> %arg0_int32x4_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmulQ_lanes32:
; CHECK: vmul.i32 q0, q0, d2[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i32>> [#uses$
  %1 = mul <4 x i32> %0, %arg0_int32x4_t          ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define <8 x i16> @vmulls8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmulls8:
;CHECK: vmull.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = sext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = sext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = mul <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <8 x i16> @vmulls8_int(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmulls8_int:
;CHECK: vmull.s8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmulls.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmulls16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmulls16:
;CHECK: vmull.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = sext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = sext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = mul <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x i32> @vmulls16_int(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmulls16_int:
;CHECK: vmull.s16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vmulls32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmulls32:
;CHECK: vmull.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = sext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = sext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = mul <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

define <2 x i64> @vmulls32_int(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmulls32_int:
;CHECK: vmull.s32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vmullu8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmullu8:
;CHECK: vmull.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = zext <8 x i8> %tmp1 to <8 x i16>
	%tmp4 = zext <8 x i8> %tmp2 to <8 x i16>
	%tmp5 = mul <8 x i16> %tmp3, %tmp4
	ret <8 x i16> %tmp5
}

define <8 x i16> @vmullu8_int(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmullu8_int:
;CHECK: vmull.u8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmullu.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define <4 x i32> @vmullu16(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmullu16:
;CHECK: vmull.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = zext <4 x i16> %tmp1 to <4 x i32>
	%tmp4 = zext <4 x i16> %tmp2 to <4 x i32>
	%tmp5 = mul <4 x i32> %tmp3, %tmp4
	ret <4 x i32> %tmp5
}

define <4 x i32> @vmullu16_int(<4 x i16>* %A, <4 x i16>* %B) nounwind {
;CHECK: vmullu16_int:
;CHECK: vmull.u16
	%tmp1 = load <4 x i16>* %A
	%tmp2 = load <4 x i16>* %B
	%tmp3 = call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %tmp1, <4 x i16> %tmp2)
	ret <4 x i32> %tmp3
}

define <2 x i64> @vmullu32(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmullu32:
;CHECK: vmull.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = zext <2 x i32> %tmp1 to <2 x i64>
	%tmp4 = zext <2 x i32> %tmp2 to <2 x i64>
	%tmp5 = mul <2 x i64> %tmp3, %tmp4
	ret <2 x i64> %tmp5
}

define <2 x i64> @vmullu32_int(<2 x i32>* %A, <2 x i32>* %B) nounwind {
;CHECK: vmullu32_int:
;CHECK: vmull.u32
	%tmp1 = load <2 x i32>* %A
	%tmp2 = load <2 x i32>* %B
	%tmp3 = call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %tmp1, <2 x i32> %tmp2)
	ret <2 x i64> %tmp3
}

define <8 x i16> @vmullp8(<8 x i8>* %A, <8 x i8>* %B) nounwind {
;CHECK: vmullp8:
;CHECK: vmull.p8
	%tmp1 = load <8 x i8>* %A
	%tmp2 = load <8 x i8>* %B
	%tmp3 = call <8 x i16> @llvm.arm.neon.vmullp.v8i16(<8 x i8> %tmp1, <8 x i8> %tmp2)
	ret <8 x i16> %tmp3
}

define arm_aapcs_vfpcc <4 x i32> @test_vmull_lanes16(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmull_lanes16
; CHECK: vmull.s16 q0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = sext <4 x i16> %arg0_int16x4_t to <4 x i32>
  %2 = sext <4 x i16> %0 to <4 x i32>
  %3 = mul <4 x i32> %1, %2
  ret <4 x i32> %3
}

define arm_aapcs_vfpcc <4 x i32> @test_vmull_lanes16_int(<4 x i16> %arg0_int16x4_t, <4 x i16> %arg1_int16x4_t) nounwind readnone {
entry:
; CHECK: test_vmull_lanes16_int
; CHECK: vmull.s16 q0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_int16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16> %arg0_int16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vmull_lanes32(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmull_lanes32
; CHECK: vmull.s32 q0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = sext <2 x i32> %arg0_int32x2_t to <2 x i64>
  %2 = sext <2 x i32> %0 to <2 x i64>
  %3 = mul <2 x i64> %1, %2
  ret <2 x i64> %3
}

define arm_aapcs_vfpcc <2 x i64> @test_vmull_lanes32_int(<2 x i32> %arg0_int32x2_t, <2 x i32> %arg1_int32x2_t) nounwind readnone {
entry:
; CHECK: test_vmull_lanes32_int
; CHECK: vmull.s32 q0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_int32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32> %arg0_int32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

define arm_aapcs_vfpcc <4 x i32> @test_vmull_laneu16(<4 x i16> %arg0_uint16x4_t, <4 x i16> %arg1_uint16x4_t) nounwind readnone {
entry:
; CHECK: test_vmull_laneu16
; CHECK: vmull.u16 q0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_uint16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = zext <4 x i16> %arg0_uint16x4_t to <4 x i32>
  %2 = zext <4 x i16> %0 to <4 x i32>
  %3 = mul <4 x i32> %1, %2
  ret <4 x i32> %3
}

define arm_aapcs_vfpcc <4 x i32> @test_vmull_laneu16_int(<4 x i16> %arg0_uint16x4_t, <4 x i16> %arg1_uint16x4_t) nounwind readnone {
entry:
; CHECK: test_vmull_laneu16_int
; CHECK: vmull.u16 q0, d0, d1[1]
  %0 = shufflevector <4 x i16> %arg1_uint16x4_t, <4 x i16> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1> ; <<4 x i16>> [#uses=1]
  %1 = tail call <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16> %arg0_uint16x4_t, <4 x i16> %0) ; <<4 x i32>> [#uses=1]
  ret <4 x i32> %1
}

define arm_aapcs_vfpcc <2 x i64> @test_vmull_laneu32(<2 x i32> %arg0_uint32x2_t, <2 x i32> %arg1_uint32x2_t) nounwind readnone {
entry:
; CHECK: test_vmull_laneu32
; CHECK: vmull.u32 q0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_uint32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = zext <2 x i32> %arg0_uint32x2_t to <2 x i64>
  %2 = zext <2 x i32> %0 to <2 x i64>
  %3 = mul <2 x i64> %1, %2
  ret <2 x i64> %3
}

define arm_aapcs_vfpcc <2 x i64> @test_vmull_laneu32_int(<2 x i32> %arg0_uint32x2_t, <2 x i32> %arg1_uint32x2_t) nounwind readnone {
entry:
; CHECK: test_vmull_laneu32_int
; CHECK: vmull.u32 q0, d0, d1[1]
  %0 = shufflevector <2 x i32> %arg1_uint32x2_t, <2 x i32> undef, <2 x i32> <i32 1, i32 1> ; <<2 x i32>> [#uses=1]
  %1 = tail call <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32> %arg0_uint32x2_t, <2 x i32> %0) ; <<2 x i64>> [#uses=1]
  ret <2 x i64> %1
}

declare <8 x i16> @llvm.arm.neon.vmulls.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmulls.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmulls.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16> @llvm.arm.neon.vmullu.v8i16(<8 x i8>, <8 x i8>) nounwind readnone
declare <4 x i32> @llvm.arm.neon.vmullu.v4i32(<4 x i16>, <4 x i16>) nounwind readnone
declare <2 x i64> @llvm.arm.neon.vmullu.v2i64(<2 x i32>, <2 x i32>) nounwind readnone

declare <8 x i16>  @llvm.arm.neon.vmullp.v8i16(<8 x i8>, <8 x i8>) nounwind readnone


; Radar 8687140
; VMULL needs to recognize BUILD_VECTORs with sign/zero-extended elements.

define <8 x i16> @vmull_extvec_s8(<8 x i8> %arg) nounwind {
; CHECK: vmull_extvec_s8
; CHECK: vmull.s8
  %tmp3 = sext <8 x i8> %arg to <8 x i16>
  %tmp4 = mul <8 x i16> %tmp3, <i16 -12, i16 -12, i16 -12, i16 -12, i16 -12, i16 -12, i16 -12, i16 -12>
  ret <8 x i16> %tmp4
}

define <8 x i16> @vmull_extvec_u8(<8 x i8> %arg) nounwind {
; CHECK: vmull_extvec_u8
; CHECK: vmull.u8
  %tmp3 = zext <8 x i8> %arg to <8 x i16>
  %tmp4 = mul <8 x i16> %tmp3, <i16 12, i16 12, i16 12, i16 12, i16 12, i16 12, i16 12, i16 12>
  ret <8 x i16> %tmp4
}

define <8 x i16> @vmull_noextvec_s8(<8 x i8> %arg) nounwind {
; Do not use VMULL if the BUILD_VECTOR element values are too big.
; CHECK: vmull_noextvec_s8
; CHECK: vmovl.s8
; CHECK: vmul.i16
  %tmp3 = sext <8 x i8> %arg to <8 x i16>
  %tmp4 = mul <8 x i16> %tmp3, <i16 -999, i16 -999, i16 -999, i16 -999, i16 -999, i16 -999, i16 -999, i16 -999>
  ret <8 x i16> %tmp4
}

define <8 x i16> @vmull_noextvec_u8(<8 x i8> %arg) nounwind {
; Do not use VMULL if the BUILD_VECTOR element values are too big.
; CHECK: vmull_noextvec_u8
; CHECK: vmovl.u8
; CHECK: vmul.i16
  %tmp3 = zext <8 x i8> %arg to <8 x i16>
  %tmp4 = mul <8 x i16> %tmp3, <i16 999, i16 999, i16 999, i16 999, i16 999, i16 999, i16 999, i16 999>
  ret <8 x i16> %tmp4
}

define <4 x i32> @vmull_extvec_s16(<4 x i16> %arg) nounwind {
; CHECK: vmull_extvec_s16
; CHECK: vmull.s16
  %tmp3 = sext <4 x i16> %arg to <4 x i32>
  %tmp4 = mul <4 x i32> %tmp3, <i32 -12, i32 -12, i32 -12, i32 -12>
  ret <4 x i32> %tmp4
}

define <4 x i32> @vmull_extvec_u16(<4 x i16> %arg) nounwind {
; CHECK: vmull_extvec_u16
; CHECK: vmull.u16
  %tmp3 = zext <4 x i16> %arg to <4 x i32>
  %tmp4 = mul <4 x i32> %tmp3, <i32 1234, i32 1234, i32 1234, i32 1234>
  ret <4 x i32> %tmp4
}

define <2 x i64> @vmull_extvec_s32(<2 x i32> %arg) nounwind {
; CHECK: vmull_extvec_s32
; CHECK: vmull.s32
  %tmp3 = sext <2 x i32> %arg to <2 x i64>
  %tmp4 = mul <2 x i64> %tmp3, <i64 -1234, i64 -1234>
  ret <2 x i64> %tmp4
}

define <2 x i64> @vmull_extvec_u32(<2 x i32> %arg) nounwind {
; CHECK: vmull_extvec_u32
; CHECK: vmull.u32
  %tmp3 = zext <2 x i32> %arg to <2 x i64>
  %tmp4 = mul <2 x i64> %tmp3, <i64 1234, i64 1234>
  ret <2 x i64> %tmp4
}

; rdar://9197392
define void @distribute(i16* %dst, i8* %src, i32 %mul) nounwind {
entry:
; CHECK: distribute:
; CHECK: vmull.u8 [[REG1:(q[0-9]+)]], d{{.*}}, [[REG2:(d[0-9]+)]]
; CHECK: vmlal.u8 [[REG1]], d{{.*}}, [[REG2]]
  %0 = trunc i32 %mul to i8
  %1 = insertelement <8 x i8> undef, i8 %0, i32 0
  %2 = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %3 = tail call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %src, i32 1)
  %4 = bitcast <16 x i8> %3 to <2 x double>
  %5 = extractelement <2 x double> %4, i32 1
  %6 = bitcast double %5 to <8 x i8>
  %7 = zext <8 x i8> %6 to <8 x i16>
  %8 = zext <8 x i8> %2 to <8 x i16>
  %9 = extractelement <2 x double> %4, i32 0
  %10 = bitcast double %9 to <8 x i8>
  %11 = zext <8 x i8> %10 to <8 x i16>
  %12 = add <8 x i16> %7, %11
  %13 = mul <8 x i16> %12, %8
  %14 = bitcast i16* %dst to i8*
  tail call void @llvm.arm.neon.vst1.v8i16(i8* %14, <8 x i16> %13, i32 2)
  ret void
}

declare <16 x i8> @llvm.arm.neon.vld1.v16i8(i8*, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.v8i16(i8*, <8 x i16>, i32) nounwind

; Take advantage of the Cortex-A8 multiplier accumulator forward.

%struct.uint8x8_t = type { <8 x i8> }

define void @distribute2(%struct.uint8x8_t* nocapture %dst, i8* %src, i32 %mul) nounwind {
entry:
; CHECK: distribute2
; CHECK-NOT: vadd.i8
; CHECK: vmul.i8
; CHECK: vmla.i8
  %0 = trunc i32 %mul to i8
  %1 = insertelement <8 x i8> undef, i8 %0, i32 0
  %2 = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %3 = tail call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %src, i32 1)
  %4 = bitcast <16 x i8> %3 to <2 x double>
  %5 = extractelement <2 x double> %4, i32 1
  %6 = bitcast double %5 to <8 x i8>
  %7 = extractelement <2 x double> %4, i32 0
  %8 = bitcast double %7 to <8 x i8>
  %9 = add <8 x i8> %6, %8
  %10 = mul <8 x i8> %9, %2
  %11 = getelementptr inbounds %struct.uint8x8_t* %dst, i32 0, i32 0
  store <8 x i8> %10, <8 x i8>* %11, align 8
  ret void
}

define void @distribute2_commutative(%struct.uint8x8_t* nocapture %dst, i8* %src, i32 %mul) nounwind {
entry:
; CHECK: distribute2_commutative
; CHECK-NOT: vadd.i8
; CHECK: vmul.i8
; CHECK: vmla.i8
  %0 = trunc i32 %mul to i8
  %1 = insertelement <8 x i8> undef, i8 %0, i32 0
  %2 = shufflevector <8 x i8> %1, <8 x i8> undef, <8 x i32> zeroinitializer
  %3 = tail call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* %src, i32 1)
  %4 = bitcast <16 x i8> %3 to <2 x double>
  %5 = extractelement <2 x double> %4, i32 1
  %6 = bitcast double %5 to <8 x i8>
  %7 = extractelement <2 x double> %4, i32 0
  %8 = bitcast double %7 to <8 x i8>
  %9 = add <8 x i8> %6, %8
  %10 = mul <8 x i8> %2, %9
  %11 = getelementptr inbounds %struct.uint8x8_t* %dst, i32 0, i32 0
  store <8 x i8> %10, <8 x i8>* %11, align 8
  ret void
}

; If one operand has a zero-extend and the other a sign-extend, vmull
; cannot be used.
define i16 @vmullWithInconsistentExtensions(<8 x i8> %vec) {
; CHECK: vmullWithInconsistentExtensions
; CHECK-NOT: vmull.s8
  %1 = sext <8 x i8> %vec to <8 x i16>
  %2 = mul <8 x i16> %1, <i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255, i16 255>
  %3 = extractelement <8 x i16> %2, i32 0
  ret i16 %3
}

; A constant build_vector created for a vmull with half-width elements must
; not introduce illegal types. <rdar://problem/11324364>
define void @vmull_buildvector() nounwind optsize ssp align 2 {
; CHECK: vmull_buildvector
entry:
  br i1 undef, label %for.end179, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.cond.loopexit:                                ; preds = %for.body33, %for.body
  br i1 undef, label %for.end179, label %for.body

for.body:                                         ; preds = %for.cond.loopexit, %for.body.lr.ph
  br i1 undef, label %for.cond.loopexit, label %for.body33.lr.ph

for.body33.lr.ph:                                 ; preds = %for.body
  %.sub = select i1 undef, i32 0, i32 undef
  br label %for.body33

for.body33:                                       ; preds = %for.body33, %for.body33.lr.ph
  %add45 = add i32 undef, undef
  %vld155 = tail call <16 x i8> @llvm.arm.neon.vld1.v16i8(i8* undef, i32 1)
  %0 = load i32** undef, align 4
  %shuffle.i250 = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> zeroinitializer
  %1 = bitcast <1 x i64> %shuffle.i250 to <8 x i8>
  %vmovl.i249 = zext <8 x i8> %1 to <8 x i16>
  %shuffle.i246 = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> zeroinitializer
  %shuffle.i240 = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> <i32 1>
  %2 = bitcast <1 x i64> %shuffle.i240 to <8 x i8>
  %3 = bitcast <16 x i8> undef to <2 x i64>
  %vmovl.i237 = zext <8 x i8> undef to <8 x i16>
  %shuffle.i234 = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> zeroinitializer
  %shuffle.i226 = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> zeroinitializer
  %vmovl.i225 = zext <8 x i8> undef to <8 x i16>
  %mul.i223 = mul <8 x i16> %vmovl.i249, %vmovl.i249
  %vshl_n = shl <8 x i16> %mul.i223, <i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2, i16 2>
  %vqsub2.i216 = tail call <8 x i16> @llvm.arm.neon.vqsubu.v8i16(<8 x i16> <i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256, i16 256>, <8 x i16> %vshl_n) nounwind
  %mul.i209 = mul <8 x i16> undef, <i16 80, i16 80, i16 80, i16 80, i16 80, i16 80, i16 80, i16 80>
  %vshr_n130 = lshr <8 x i16> undef, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %vshr_n134 = lshr <8 x i16> %mul.i209, <i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8, i16 8>
  %sub.i205 = sub <8 x i16> <i16 80, i16 80, i16 80, i16 80, i16 80, i16 80, i16 80, i16 80>, %vshr_n130
  %sub.i203 = sub <8 x i16> <i16 80, i16 80, i16 80, i16 80, i16 80, i16 80, i16 80, i16 80>, %vshr_n134
  %add.i200 = add <8 x i16> %sub.i205, <i16 96, i16 96, i16 96, i16 96, i16 96, i16 96, i16 96, i16 96>
  %add.i198 = add <8 x i16> %add.i200, %sub.i203
  %mul.i194 = mul <8 x i16> %add.i198, %vmovl.i237
  %mul.i191 = mul <8 x i16> %vshr_n130, undef
  %add.i192 = add <8 x i16> %mul.i191, %mul.i194
  %mul.i187 = mul <8 x i16> %vshr_n134, undef
  %add.i188 = add <8 x i16> %mul.i187, %add.i192
  %mul.i185 = mul <8 x i16> undef, undef
  %add.i186 = add <8 x i16> %mul.i185, undef
  %vrshr_n160 = tail call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %add.i188, <8 x i16> <i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8>)
  %vrshr_n163 = tail call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %add.i186, <8 x i16> <i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8, i16 -8>)
  %mul.i184 = mul <8 x i16> undef, %vrshr_n160
  %mul.i181 = mul <8 x i16> undef, %vmovl.i225
  %add.i182 = add <8 x i16> %mul.i181, %mul.i184
  %vrshr_n170 = tail call <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16> %add.i182, <8 x i16> <i16 -7, i16 -7, i16 -7, i16 -7, i16 -7, i16 -7, i16 -7, i16 -7>)
  %vqmovn1.i180 = tail call <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16> %vrshr_n170) nounwind
  %4 = bitcast <8 x i8> %vqmovn1.i180 to <1 x i64>
  %shuffle.i = shufflevector <1 x i64> %4, <1 x i64> undef, <2 x i32> <i32 0, i32 1>
  %5 = bitcast <2 x i64> %shuffle.i to <16 x i8>
  store <16 x i8> %5, <16 x i8>* undef, align 16
  %add177 = add nsw i32 undef, 16
  br i1 undef, label %for.body33, label %for.cond.loopexit

for.end179:                                       ; preds = %for.cond.loopexit, %entry
  ret void
}

declare <8 x i16> @llvm.arm.neon.vrshiftu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <8 x i16> @llvm.arm.neon.vqsubu.v8i16(<8 x i16>, <8 x i16>) nounwind readnone
declare <8 x i8> @llvm.arm.neon.vqmovnu.v8i8(<8 x i16>) nounwind readnone

; vmull lowering would create a zext(v4i8 load()) instead of a zextload(v4i8),
; creating an illegal type during legalization and causing an assert.
; PR15970
define void @no_illegal_types_vmull_sext(<4 x i32> %a) {
entry:
  %wide.load283.i = load <4 x i8>* undef, align 1
  %0 = sext <4 x i8> %wide.load283.i to <4 x i32>
  %1 = sub nsw <4 x i32> %0, %a
  %2 = mul nsw <4 x i32> %1, %1
  %predphi290.v.i = select <4 x i1> undef, <4 x i32> undef, <4 x i32> %2
  store <4 x i32> %predphi290.v.i, <4 x i32>* undef, align 4
  ret void
}
define void @no_illegal_types_vmull_zext(<4 x i32> %a) {
entry:
  %wide.load283.i = load <4 x i8>* undef, align 1
  %0 = zext <4 x i8> %wide.load283.i to <4 x i32>
  %1 = sub nsw <4 x i32> %0, %a
  %2 = mul nsw <4 x i32> %1, %1
  %predphi290.v.i = select <4 x i1> undef, <4 x i32> undef, <4 x i32> %2
  store <4 x i32> %predphi290.v.i, <4 x i32>* undef, align 4
  ret void
}
