; RUN: llc < %s -mcpu=cortex-a8 -verify-machineinstrs
; PR11829
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7-none-linux-gnueabi"

define arm_aapcs_vfpcc void @foo(ptr nocapture %arg) nounwind uwtable align 2 {
bb:
  br i1 undef, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  unreachable

bb2:                                              ; preds = %bb
  br label %bb3

bb3:                                              ; preds = %bb4, %bb2
  %tmp = icmp slt i32 undef, undef
  br i1 %tmp, label %bb4, label %bb67

bb4:                                              ; preds = %bb3
  %tmp5 = load <4 x i32>, ptr undef, align 16
  %tmp6 = and <4 x i32> %tmp5, <i32 8388607, i32 8388607, i32 8388607, i32 8388607>
  %tmp7 = or <4 x i32> %tmp6, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %tmp8 = bitcast <4 x i32> %tmp7 to <4 x float>
  %or = or i128 shl (i128 zext (i64 trunc (i128 lshr (i128 bitcast (<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00> to i128), i128 64) to i64) to i128), i128 64), zext (i64 trunc (i128 bitcast (<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00> to i128) to i64) to i128)
  %bc = bitcast i128 %or to <4 x float>
  %tmp9 = fsub <4 x float> %tmp8, %bc
  %tmp10 = fmul <4 x float> undef, %tmp9
  %tmp11 = fadd <4 x float> undef, %tmp10
  %tmp12 = bitcast <4 x float> zeroinitializer to i128
  %tmp13 = lshr i128 %tmp12, 64
  %tmp14 = trunc i128 %tmp13 to i64
  %tmp15 = insertvalue [2 x i64] undef, i64 %tmp14, 1
  %tmp16 = call <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float> %tmp11) nounwind
  %tmp17 = call <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float> %tmp16, <4 x float> %tmp11) nounwind
  %tmp18 = fmul <4 x float> %tmp17, %tmp16
  %tmp19 = call <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float> %tmp18, <4 x float> %tmp11) nounwind
  %tmp20 = fmul <4 x float> %tmp19, %tmp18
  %tmp21 = fmul <4 x float> %tmp20, zeroinitializer
  %tmp22 = call <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float> %tmp21, <4 x float> undef) nounwind
  call arm_aapcs_vfpcc  void @bar(ptr null, ptr undef, ptr undef, [2 x i64] zeroinitializer) nounwind
  %tmp23 = bitcast <4 x float> %tmp22 to i128
  %tmp24 = trunc i128 %tmp23 to i64
  %tmp25 = insertvalue [2 x i64] undef, i64 %tmp24, 0
  %tmp26 = insertvalue [2 x i64] %tmp25, i64 0, 1
  %tmp27 = load float, ptr undef, align 4
  %tmp28 = insertelement <4 x float> undef, float %tmp27, i32 3
  %tmp29 = load <4 x i32>, ptr undef, align 16
  %tmp30 = and <4 x i32> %tmp29, <i32 8388607, i32 8388607, i32 8388607, i32 8388607>
  %tmp31 = or <4 x i32> %tmp30, <i32 1065353216, i32 1065353216, i32 1065353216, i32 1065353216>
  %tmp32 = bitcast <4 x i32> %tmp31 to <4 x float>
  %or2 = or i128 shl (i128 zext (i64 trunc (i128 lshr (i128 bitcast (<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00> to i128), i128 64) to i64) to i128), i128 64), zext (i64 trunc (i128 bitcast (<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00> to i128) to i64) to i128)
  %bc2 = bitcast i128 %or2 to <4 x float>
  %tmp33 = fsub <4 x float> %tmp32, %bc2
  %tmp34 = call <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float> undef, <4 x float> %tmp28) nounwind
  %tmp35 = fmul <4 x float> %tmp34, undef
  %tmp36 = fmul <4 x float> %tmp35, undef
  %tmp37 = call arm_aapcs_vfpcc  ptr undef(ptr undef) nounwind
  %tmp38 = load float, ptr undef, align 4
  %tmp39 = insertelement <2 x float> undef, float %tmp38, i32 0
  %tmp40 = call arm_aapcs_vfpcc  ptr undef(ptr undef) nounwind
  %tmp41 = load float, ptr undef, align 4
  %tmp42 = insertelement <4 x float> undef, float %tmp41, i32 3
  %tmp43 = shufflevector <2 x float> %tmp39, <2 x float> undef, <4 x i32> zeroinitializer
  %tmp44 = fmul <4 x float> %tmp33, %tmp43
  %tmp45 = fadd <4 x float> %tmp42, %tmp44
  %tmp46 = fsub <4 x float> %tmp45, undef
  %tmp47 = fmul <4 x float> %tmp46, %tmp36
  %tmp48 = fadd <4 x float> undef, %tmp47
  %tmp49 = call arm_aapcs_vfpcc  ptr undef(ptr undef) nounwind
  %tmp50 = load float, ptr undef, align 4
  %tmp51 = insertelement <4 x float> undef, float %tmp50, i32 3
  %tmp52 = call arm_aapcs_vfpcc ptr null(ptr undef) nounwind
  %tmp54 = load float, ptr %tmp52, align 4
  %tmp55 = insertelement <4 x float> undef, float %tmp54, i32 3
  %tmp56 = fsub <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %tmp22
  %tmp57 = call <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float> %tmp56, <4 x float> %tmp55) nounwind
  %tmp58 = fmul <4 x float> undef, %tmp57
  %tmp59 = fsub <4 x float> %tmp51, %tmp48
  %tmp60 = fsub <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %tmp58
  %tmp61 = fmul <4 x float> %tmp59, %tmp60
  %tmp62 = fadd <4 x float> %tmp48, %tmp61
  call arm_aapcs_vfpcc  void @baz(ptr undef, ptr undef, [2 x i64] %tmp26, ptr undef)
  %tmp63 = bitcast <4 x float> %tmp62 to i128
  %tmp64 = lshr i128 %tmp63, 64
  %tmp65 = trunc i128 %tmp64 to i64
  %tmp66 = insertvalue [2 x i64] zeroinitializer, i64 %tmp65, 1
  call arm_aapcs_vfpcc  void @quux(ptr undef, ptr undef, [2 x i64] undef, ptr undef, [2 x i64] %tmp66, ptr undef, ptr undef, [2 x i64] %tmp26, [2 x i64] %tmp15, ptr undef)
  br label %bb3

bb67:                                             ; preds = %bb3
  ret void
}

declare arm_aapcs_vfpcc void @bar(ptr, ptr, ptr, [2 x i64])

declare arm_aapcs_vfpcc void @baz(ptr, ptr nocapture, [2 x i64], ptr nocapture) nounwind uwtable inlinehint align 2

declare arm_aapcs_vfpcc void @quux(ptr, ptr, [2 x i64], ptr nocapture, [2 x i64], ptr nocapture, ptr nocapture, [2 x i64], [2 x i64], ptr nocapture) nounwind uwtable inlinehint align 2

declare <4 x float> @llvm.arm.neon.vmins.v4f32(<4 x float>, <4 x float>) nounwind readnone

declare <4 x float> @llvm.arm.neon.vrecps.v4f32(<4 x float>, <4 x float>) nounwind readnone

declare <4 x float> @llvm.arm.neon.vrecpe.v4f32(<4 x float>) nounwind readnone
