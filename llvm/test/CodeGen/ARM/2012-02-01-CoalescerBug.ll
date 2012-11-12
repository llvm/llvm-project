; RUN: llc -verify-coalescing < %s
; PR11868

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:64:128-a0:0:64-n32-S64"
target triple = "armv7-none-linux-gnueabi"

%0 = type { <4 x float> }
%1 = type { <4 x float> }

@foo = external global %0, align 16

define arm_aapcs_vfpcc void @bar(float, i1 zeroext, i1 zeroext) nounwind {
  %4 = load <4 x float>* getelementptr inbounds (%0* @foo, i32 0, i32 0), align 16
  %5 = extractelement <4 x float> %4, i32 0
  %6 = extractelement <4 x float> %4, i32 1
  %7 = extractelement <4 x float> %4, i32 2
  %8 = insertelement <4 x float> undef, float %5, i32 0
  %9 = insertelement <4 x float> %8, float %6, i32 1
  %10 = insertelement <4 x float> %9, float %7, i32 2
  %11 = insertelement <4 x float> %10, float 0.000000e+00, i32 3
  store <4 x float> %11, <4 x float>* undef, align 16 
  call arm_aapcs_vfpcc  void @baz(%1* undef, float 0.000000e+00) nounwind
  ret void
}

declare arm_aapcs_vfpcc void @baz(%1*, float)
