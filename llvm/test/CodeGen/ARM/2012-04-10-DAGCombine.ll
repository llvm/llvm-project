; RUN: llc < %s -march=arm -mcpu=cortex-a9 -enable-unsafe-fp-math
;target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
;target triple = "armv7-none-linux-gnueabi"

define arm_aapcs_vfpcc void @foo(<4 x float> %arg) nounwind align 2 {
bb4:
  %tmp = extractelement <2 x float> undef, i32 0
  br i1 undef, label %bb18, label %bb5

bb5:                                              ; preds = %bb4
  %tmp6 = fadd float %tmp, -1.500000e+01
  %tmp7 = fdiv float %tmp6, 2.000000e+01
  %tmp8 = fadd float %tmp7, 1.000000e+00
  %tmp9 = fdiv float 1.000000e+00, %tmp8
  %tmp10 = fsub float 1.000000e+00, %tmp9
  %tmp11 = fmul float %tmp10, 1.000000e+01
  %tmp12 = fadd float %tmp11, 1.500000e+01
  %tmp13 = fdiv float %tmp12, %tmp
  %tmp14 = insertelement <2 x float> undef, float %tmp13, i32 0
  %tmp15 = shufflevector <2 x float> %tmp14, <2 x float> undef, <4 x i32> zeroinitializer
  %tmp16 = fmul <4 x float> zeroinitializer, %tmp15
  %tmp17 = fadd <4 x float> %tmp16, %arg
  store <4 x float> %tmp17, <4 x float>* undef, align 8, !tbaa !0
  br label %bb18

bb18:                                             ; preds = %bb5, %bb4
  ret void
}

!0 = metadata !{metadata !"omnipotent char", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA", null}
