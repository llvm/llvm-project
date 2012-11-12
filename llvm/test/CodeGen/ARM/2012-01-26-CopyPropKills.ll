; RUN: llc < %s -mcpu=cortex-a9 -join-liveintervals=0 -verify-machineinstrs
; PR11765
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7-none-linux-gnueabi"

; This test case exercises the MachineCopyPropagation pass by disabling the
; RegisterCoalescer.

define arm_aapcs_vfpcc void @foo(i8* %arg) nounwind uwtable align 2 {
bb:
  br i1 undef, label %bb1, label %bb2

bb1:                                              ; preds = %bb
  unreachable

bb2:                                              ; preds = %bb
  br i1 undef, label %bb92, label %bb3

bb3:                                              ; preds = %bb2
  %tmp = or <4 x i32> undef, undef
  %tmp4 = bitcast <4 x i32> %tmp to <4 x float>
  %tmp5 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %tmp4
  %tmp6 = bitcast <4 x i32> zeroinitializer to <4 x float>
  %tmp7 = fmul <4 x float> %tmp6, <float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>
  %tmp8 = bitcast <4 x float> %tmp7 to <2 x i64>
  %tmp9 = shufflevector <2 x i64> %tmp8, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp10 = bitcast <1 x i64> %tmp9 to <2 x float>
  %tmp11 = shufflevector <2 x i64> %tmp8, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp12 = bitcast <1 x i64> %tmp11 to <2 x float>
  %tmp13 = shufflevector <2 x float> %tmp10, <2 x float> %tmp12, <2 x i32> <i32 0, i32 2>
  %tmp14 = shufflevector <2 x float> %tmp10, <2 x float> undef, <2 x i32> <i32 1, i32 2>
  %tmp15 = bitcast <2 x float> %tmp14 to <1 x i64>
  %tmp16 = bitcast <4 x i32> zeroinitializer to <2 x i64>
  %tmp17 = shufflevector <2 x i64> %tmp16, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp18 = bitcast <1 x i64> %tmp17 to <2 x i32>
  %tmp19 = and <2 x i32> %tmp18, <i32 -1, i32 0>
  %tmp20 = bitcast <2 x float> %tmp13 to <2 x i32>
  %tmp21 = and <2 x i32> %tmp20, <i32 0, i32 -1>
  %tmp22 = or <2 x i32> %tmp19, %tmp21
  %tmp23 = bitcast <2 x i32> %tmp22 to <1 x i64>
  %tmp24 = shufflevector <1 x i64> %tmp23, <1 x i64> undef, <2 x i32> <i32 0, i32 1>
  %tmp25 = bitcast <2 x i64> %tmp24 to <4 x float>
  %tmp26 = shufflevector <2 x i64> %tmp16, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp27 = bitcast <1 x i64> %tmp26 to <2 x i32>
  %tmp28 = and <2 x i32> %tmp27, <i32 -1, i32 0>
  %tmp29 = and <2 x i32> undef, <i32 0, i32 -1>
  %tmp30 = or <2 x i32> %tmp28, %tmp29
  %tmp31 = bitcast <2 x i32> %tmp30 to <1 x i64>
  %tmp32 = insertelement <4 x float> %tmp25, float 0.000000e+00, i32 3
  %tmp33 = fmul <4 x float> undef, <float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>
  %tmp34 = fadd <4 x float> %tmp33, %tmp32
  %tmp35 = fmul <4 x float> %tmp33, zeroinitializer
  %tmp36 = fadd <4 x float> %tmp35, zeroinitializer
  %tmp37 = fadd <4 x float> %tmp35, zeroinitializer
  %tmp38 = bitcast <4 x float> %tmp34 to <2 x i64>
  %tmp39 = shufflevector <2 x i64> %tmp38, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp40 = bitcast <1 x i64> %tmp39 to <2 x float>
  %tmp41 = shufflevector <2 x float> %tmp40, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %tmp42 = load <4 x float>* null, align 16, !tbaa !0
  %tmp43 = fmul <4 x float> %tmp42, %tmp41
  %tmp44 = load <4 x float>* undef, align 16, !tbaa !0
  %tmp45 = fadd <4 x float> undef, %tmp43
  %tmp46 = fadd <4 x float> undef, %tmp45
  %tmp47 = bitcast <4 x float> %tmp36 to <2 x i64>
  %tmp48 = shufflevector <2 x i64> %tmp47, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp49 = bitcast <1 x i64> %tmp48 to <2 x float>
  %tmp50 = shufflevector <2 x float> %tmp49, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %tmp51 = fmul <4 x float> %tmp42, %tmp50
  %tmp52 = fmul <4 x float> %tmp44, undef
  %tmp53 = fadd <4 x float> %tmp52, %tmp51
  %tmp54 = fadd <4 x float> undef, %tmp53
  %tmp55 = bitcast <4 x float> %tmp37 to <2 x i64>
  %tmp56 = shufflevector <2 x i64> %tmp55, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp57 = bitcast <1 x i64> %tmp56 to <2 x float>
  %tmp58 = shufflevector <2 x float> %tmp57, <2 x float> undef, <4 x i32> zeroinitializer
  %tmp59 = fmul <4 x float> undef, %tmp58
  %tmp60 = fadd <4 x float> %tmp59, undef
  %tmp61 = fadd <4 x float> %tmp60, zeroinitializer
  %tmp62 = load void (i8*, i8*)** undef, align 4
  call arm_aapcs_vfpcc  void %tmp62(i8* sret undef, i8* undef) nounwind
  %tmp63 = bitcast <4 x float> %tmp46 to i128
  %tmp64 = bitcast <4 x float> %tmp54 to i128
  %tmp65 = bitcast <4 x float> %tmp61 to i128
  %tmp66 = lshr i128 %tmp63, 64
  %tmp67 = trunc i128 %tmp66 to i64
  %tmp68 = insertvalue [8 x i64] undef, i64 %tmp67, 1
  %tmp69 = insertvalue [8 x i64] %tmp68, i64 undef, 2
  %tmp70 = lshr i128 %tmp64, 64
  %tmp71 = trunc i128 %tmp70 to i64
  %tmp72 = insertvalue [8 x i64] %tmp69, i64 %tmp71, 3
  %tmp73 = trunc i128 %tmp65 to i64
  %tmp74 = insertvalue [8 x i64] %tmp72, i64 %tmp73, 4
  %tmp75 = insertvalue [8 x i64] %tmp74, i64 undef, 5
  %tmp76 = insertvalue [8 x i64] %tmp75, i64 undef, 6
  %tmp77 = insertvalue [8 x i64] %tmp76, i64 undef, 7
  call arm_aapcs_vfpcc  void @bar(i8* sret null, [8 x i64] %tmp77) nounwind
  %tmp78 = call arm_aapcs_vfpcc  i8* null(i8* null) nounwind
  %tmp79 = bitcast i8* %tmp78 to i512*
  %tmp80 = load i512* %tmp79, align 16
  %tmp81 = lshr i512 %tmp80, 128
  %tmp82 = trunc i512 %tmp80 to i128
  %tmp83 = trunc i512 %tmp81 to i128
  %tmp84 = bitcast i128 %tmp83 to <4 x float>
  %tmp85 = bitcast <4 x float> %tmp84 to <2 x i64>
  %tmp86 = shufflevector <2 x i64> %tmp85, <2 x i64> undef, <1 x i32> <i32 1>
  %tmp87 = bitcast <1 x i64> %tmp86 to <2 x float>
  %tmp88 = shufflevector <2 x float> %tmp87, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %tmp89 = fmul <4 x float> undef, %tmp88
  %tmp90 = fadd <4 x float> %tmp89, undef
  %tmp91 = fadd <4 x float> undef, %tmp90
  store <4 x float> %tmp91, <4 x float>* undef, align 16, !tbaa !0
  unreachable

bb92:                                             ; preds = %bb2
  ret void
}

declare arm_aapcs_vfpcc void @bar(i8* noalias nocapture sret, [8 x i64]) nounwind uwtable inlinehint

!0 = metadata !{metadata !"omnipotent char", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA", null}
