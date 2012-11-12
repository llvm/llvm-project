; RUN: llc < %s
; PR11861
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7-none-linux-gnueabi"

define arm_aapcs_vfpcc void @foo() nounwind align 2 {
  br i1 undef, label %5, label %1

; <label>:1                                       ; preds = %0
  %2 = shufflevector <1 x i64> zeroinitializer, <1 x i64> undef, <2 x i32> <i32 0, i32 1>
  %3 = bitcast <2 x i64> %2 to <4 x float>
  store <4 x float> zeroinitializer, <4 x float>* undef, align 16, !tbaa !0
  store <4 x float> zeroinitializer, <4 x float>* undef, align 16, !tbaa !0
  store <4 x float> %3, <4 x float>* undef, align 16, !tbaa !0
  %4 = insertelement <4 x float> %3, float 8.000000e+00, i32 2
  store <4 x float> %4, <4 x float>* undef, align 16, !tbaa !0
  unreachable

; <label>:5                                       ; preds = %0
  ret void
}

!0 = metadata !{metadata !"omnipotent char", metadata !1}
!1 = metadata !{metadata !"Simple C/C++ TBAA", null}
