; RUN: llc -O=0 -verify-machineinstrs -mtriple aarch64--- \
; RUN: -stop-before=instruction-select -global-isel %s -o - | FileCheck %s

; Make sure that we choose a FPR for the G_FCEIL and G_LOAD instead of a GPR.

declare float @llvm.ceil.f32(float)

; CHECK-LABEL: name:            foo
define float @foo(float) {
  store float %0, float* undef, align 4
  ; CHECK: %2:fpr(s32) = G_LOAD %1(p0)
  ; CHECK-NEXT: %3:fpr(s32) = G_FCEIL %2
  %2 = load float, float* undef, align 4
  %3 = call float @llvm.ceil.f32(float %2)
  ret float %3
}
