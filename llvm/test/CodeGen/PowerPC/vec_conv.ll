; RUN: llc -verify-machineinstrs -mattr=+altivec < %s | FileCheck %s

; Check vector float/int conversion using altivec.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@cte_float = global <4 x float> <float 6.5e+00, float 6.5e+00, float 6.5e+00, float 6.5e+00>, align 16
@cte_int = global <4 x i32> <i32 6, i32 6, i32 6, i32 6>, align 16


define void @v4f32_to_v4i32(<4 x float> %x, ptr nocapture %y) nounwind {
entry:
  %0 = load <4 x float>, ptr @cte_float, align 16
  %mul = fmul <4 x float> %0, %x
  %1 = fptosi <4 x float> %mul to <4 x i32>
  store <4 x i32> %1, ptr %y, align 16
  ret void
}
;CHECK-LABEL: v4f32_to_v4i32:
;CHECK: vctsxs {{[0-9]+}}, {{[0-9]+}}, 0


define void @v4f32_to_v4u32(<4 x float> %x, ptr nocapture %y) nounwind {
entry:
  %0 = load <4 x float>, ptr @cte_float, align 16
  %mul = fmul <4 x float> %0, %x
  %1 = fptoui <4 x float> %mul to <4 x i32>
  store <4 x i32> %1, ptr %y, align 16
  ret void
}
;CHECK-LABEL: v4f32_to_v4u32:
;CHECK: vctuxs {{[0-9]+}}, {{[0-9]+}}, 0


define void @v4i32_to_v4f32(<4 x i32> %x, ptr nocapture %y) nounwind {
entry:
  %0 = load <4 x i32>, ptr @cte_int, align 16
  %mul = mul <4 x i32> %0, %x
  %1 = sitofp <4 x i32> %mul to <4 x float>
  store <4 x float> %1, ptr %y, align 16
  ret void
}
;CHECK-LABEL: v4i32_to_v4f32:
;CHECK: vcfsx {{[0-9]+}}, {{[0-9]+}}, 0


define void @v4u32_to_v4f32(<4 x i32> %x, ptr nocapture %y) nounwind {
entry:
  %0 = load <4 x i32>, ptr @cte_int, align 16
  %mul = mul <4 x i32> %0, %x
  %1 = uitofp <4 x i32> %mul to <4 x float>
  store <4 x float> %1, ptr %y, align 16
  ret void
}
;CHECK-LABEL: v4u32_to_v4f32:
;CHECK: vcfux {{[0-9]+}}, {{[0-9]+}}, 0
