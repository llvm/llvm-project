; RUN: llc -mcpu=cortex-a8 -mattr=+neon < %s | grep vneg
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "armv7-eabi"

%aaa = type { %fff, %fff }
%bbb = type { [6 x %ddd] }
%ccc = type { %eee, %fff }
%ddd = type { %fff }
%eee = type { %fff, %fff, %fff, %fff }
%fff = type { %struct.vec_float4 }
%struct.vec_float4 = type { <4 x float> }

define linkonce_odr arm_aapcs_vfpcc void @foo(%eee* noalias sret %agg.result, i64 %tfrm.0.0, i64 %tfrm.0.1, i64 %tfrm.0.2, i64 %tfrm.0.3, i64 %tfrm.0.4, i64 %tfrm.0.5, i64 %tfrm.0.6, i64 %tfrm.0.7) nounwind noinline {
entry:
  %tmp104 = zext i64 %tfrm.0.2 to i512            ; <i512> [#uses=1]
  %tmp105 = shl i512 %tmp104, 128                 ; <i512> [#uses=1]
  %tmp118 = zext i64 %tfrm.0.3 to i512            ; <i512> [#uses=1]
  %tmp119 = shl i512 %tmp118, 192                 ; <i512> [#uses=1]
  %ins121 = or i512 %tmp119, %tmp105              ; <i512> [#uses=1]
  %tmp99 = zext i64 %tfrm.0.4 to i512             ; <i512> [#uses=1]
  %tmp100 = shl i512 %tmp99, 256                  ; <i512> [#uses=1]
  %tmp123 = zext i64 %tfrm.0.5 to i512            ; <i512> [#uses=1]
  %tmp124 = shl i512 %tmp123, 320                 ; <i512> [#uses=1]
  %tmp96 = zext i64 %tfrm.0.6 to i512             ; <i512> [#uses=1]
  %tmp97 = shl i512 %tmp96, 384                   ; <i512> [#uses=1]
  %tmp128 = zext i64 %tfrm.0.7 to i512            ; <i512> [#uses=1]
  %tmp129 = shl i512 %tmp128, 448                 ; <i512> [#uses=1]
  %mask.masked = or i512 %tmp124, %tmp100         ; <i512> [#uses=1]
  %ins131 = or i512 %tmp129, %tmp97               ; <i512> [#uses=1]
  %tmp109132 = zext i64 %tfrm.0.0 to i128         ; <i128> [#uses=1]
  %tmp113134 = zext i64 %tfrm.0.1 to i128         ; <i128> [#uses=1]
  %tmp114133 = shl i128 %tmp113134, 64            ; <i128> [#uses=1]
  %tmp94 = or i128 %tmp114133, %tmp109132         ; <i128> [#uses=1]
  %tmp95 = bitcast i128 %tmp94 to <4 x float>     ; <<4 x float>> [#uses=0]
  %tmp82 = lshr i512 %ins121, 128                 ; <i512> [#uses=1]
  %tmp83 = trunc i512 %tmp82 to i128              ; <i128> [#uses=1]
  %tmp84 = bitcast i128 %tmp83 to <4 x float>     ; <<4 x float>> [#uses=0]
  %tmp86 = lshr i512 %mask.masked, 256            ; <i512> [#uses=1]
  %tmp87 = trunc i512 %tmp86 to i128              ; <i128> [#uses=1]
  %tmp88 = bitcast i128 %tmp87 to <4 x float>     ; <<4 x float>> [#uses=0]
  %tmp90 = lshr i512 %ins131, 384                 ; <i512> [#uses=1]
  %tmp91 = trunc i512 %tmp90 to i128              ; <i128> [#uses=1]
  %tmp92 = bitcast i128 %tmp91 to <4 x float>     ; <<4 x float>> [#uses=1]
  %tmp = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %tmp92 ; <<4 x float>> [#uses=1]
  %tmp28 = getelementptr inbounds %eee* %agg.result, i32 0, i32 3, i32 0, i32 0 ; <<4 x float>*> [#uses=1]
  store <4 x float> %tmp, <4 x float>* %tmp28, align 16
  ret void
}
