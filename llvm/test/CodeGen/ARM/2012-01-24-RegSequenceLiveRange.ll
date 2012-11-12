; RUN: llc < %s -mcpu=cortex-a8 -verify-machineinstrs -verify-coalescing
; PR11841
; PR11829
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7-none-linux-eabi"

; This test case is exercising REG_SEQUENCE, and chains of REG_SEQUENCE.
define arm_aapcs_vfpcc void @foo(i8* nocapture %arg, i8* %arg1) nounwind align 2 {
bb:
  %tmp = load <2 x float>* undef, align 8, !tbaa !0
  %tmp2 = extractelement <2 x float> %tmp, i32 0
  %tmp3 = insertelement <4 x float> undef, float %tmp2, i32 0
  %tmp4 = insertelement <4 x float> %tmp3, float 0.000000e+00, i32 1
  %tmp5 = insertelement <4 x float> %tmp4, float 0.000000e+00, i32 2
  %tmp6 = insertelement <4 x float> %tmp5, float 0.000000e+00, i32 3
  %tmp7 = extractelement <2 x float> %tmp, i32 1
  %tmp8 = insertelement <4 x float> %tmp3, float %tmp7, i32 1
  %tmp9 = insertelement <4 x float> %tmp8, float 0.000000e+00, i32 2
  %tmp10 = insertelement <4 x float> %tmp9, float 0.000000e+00, i32 3
  %tmp11 = bitcast <4 x float> %tmp6 to <2 x i64>
  %tmp12 = shufflevector <2 x i64> %tmp11, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp13 = bitcast <1 x i64> %tmp12 to <2 x float>
  %tmp14 = shufflevector <2 x float> %tmp13, <2 x float> undef, <4 x i32> zeroinitializer
  %tmp15 = bitcast <4 x float> %tmp14 to <2 x i64>
  %tmp16 = shufflevector <2 x i64> %tmp15, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp17 = bitcast <1 x i64> %tmp16 to <2 x float>
  %tmp18 = extractelement <2 x float> %tmp17, i32 0
  tail call arm_aapcs_vfpcc  void @bar(i8* undef, float %tmp18, float undef, float 0.000000e+00) nounwind
  %tmp19 = bitcast <4 x float> %tmp10 to <2 x i64>
  %tmp20 = shufflevector <2 x i64> %tmp19, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp21 = bitcast <1 x i64> %tmp20 to <2 x float>
  %tmp22 = shufflevector <2 x float> %tmp21, <2 x float> undef, <4 x i32> <i32 1, i32 1, i32 1, i32 1>
  %tmp23 = bitcast <4 x float> %tmp22 to <2 x i64>
  %tmp24 = shufflevector <2 x i64> %tmp23, <2 x i64> undef, <1 x i32> zeroinitializer
  %tmp25 = bitcast <1 x i64> %tmp24 to <2 x float>
  %tmp26 = extractelement <2 x float> %tmp25, i32 0
  tail call arm_aapcs_vfpcc  void @bar(i8* undef, float undef, float %tmp26, float 0.000000e+00) nounwind
  ret void
}

define arm_aapcs_vfpcc void @foo2() nounwind uwtable {
entry:
  br i1 undef, label %for.end, label %cond.end295

cond.end295:                                      ; preds = %entry
  %shuffle.i39.i.i1035 = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> zeroinitializer
  %shuffle.i38.i.i1036 = shufflevector <2 x i64> zeroinitializer, <2 x i64> undef, <1 x i32> zeroinitializer
  %shuffle.i37.i.i1037 = shufflevector <1 x i64> %shuffle.i39.i.i1035, <1 x i64> %shuffle.i38.i.i1036, <2 x i32> <i32 0, i32 1>
  %0 = bitcast <2 x i64> %shuffle.i37.i.i1037 to <4 x float>
  %1 = bitcast <4 x float> undef to <2 x i64>
  %shuffle.i36.i.i = shufflevector <2 x i64> %1, <2 x i64> undef, <1 x i32> zeroinitializer
  %shuffle.i35.i.i = shufflevector <2 x i64> undef, <2 x i64> undef, <1 x i32> zeroinitializer
  %shuffle.i34.i.i = shufflevector <1 x i64> %shuffle.i36.i.i, <1 x i64> %shuffle.i35.i.i, <2 x i32> <i32 0, i32 1>
  %2 = bitcast <2 x i64> %shuffle.i34.i.i to <4 x float>
  tail call void @llvm.arm.neon.vst1.v4f32(i8* undef, <4 x float> %0, i32 4) nounwind
  tail call void @llvm.arm.neon.vst1.v4f32(i8* undef, <4 x float> %2, i32 4) nounwind
  unreachable

for.end:                                          ; preds = %entry
  ret void
}

; Check that pseudo-expansion preserves <undef> flags.
define void @foo3(i8* %p) nounwind ssp {
entry:
  tail call void @llvm.arm.neon.vst2.v4f32(i8* %p, <4 x float> undef, <4 x float> undef, i32 4)
  ret void
}

declare arm_aapcs_vfpcc void @bar(i8*, float, float, float)
declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32) nounwind
declare void @llvm.arm.neon.vst2.v4f32(i8*, <4 x float>, <4 x float>, i32) nounwind

!0 = metadata !{metadata !"omnipotent char", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA", null}
