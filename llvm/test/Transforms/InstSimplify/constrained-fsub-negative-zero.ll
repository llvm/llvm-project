; RUN: opt -passes=instsimplify -S < %s | FileCheck %s

; -0.0 - (-0.0) is negative zero under downward rounding.

; CHECK-LABEL: @subtract_down(
; CHECK: call float @llvm.experimental.constrained.fsub.f32
define float @subtract_down(float %x) strictfp {
  %negative = fneg float %x
  %answer = call float @llvm.experimental.constrained.fsub.f32(float -0.0, float %negative, metadata !"round.downward", metadata !"fpexcept.ignore") strictfp
  ret float %answer
}

; CHECK-LABEL: @subtract_dynamic(
; CHECK: call float @llvm.experimental.constrained.fsub.f32
define float @subtract_dynamic(float %x) strictfp {
  %negative = fneg float %x
  %answer = call float @llvm.experimental.constrained.fsub.f32(float -0.0, float %negative, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  ret float %answer
}

; CHECK-LABEL: @subtract_nearest(
; CHECK: ret float %x
define float @subtract_nearest(float %x) strictfp {
  %negative = fneg float %x
  %answer = call float @llvm.experimental.constrained.fsub.f32(float -0.0, float %negative, metadata !"round.tonearest", metadata !"fpexcept.ignore") strictfp
  ret float %answer
}

; CHECK-LABEL: @subtract_dynamic_nsz(
; CHECK: ret float %x
define float @subtract_dynamic_nsz(float %x) strictfp {
  %negative = fneg float %x
  %answer = call nsz float @llvm.experimental.constrained.fsub.f32(float -0.0, float %negative, metadata !"round.dynamic", metadata !"fpexcept.ignore") strictfp
  ret float %answer
}

declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
