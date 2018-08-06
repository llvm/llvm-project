; RUN: llc -mtriple=armv8 -mattr=+neon %s -o - | FileCheck %s

declare float @llvm.arm.neon.vrintn.f32(float) nounwind readnone

; CHECK-LABEL: vrintn_f32:
; CHECK: vrintn.f32
define float @vrintn_f32(float* %A) nounwind {
  %tmp1 = load float, float* %A
  %tmp2 = call float @llvm.arm.neon.vrintn.f32(float %tmp1)
  ret float %tmp2
}
