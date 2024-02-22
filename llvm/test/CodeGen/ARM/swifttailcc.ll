; RUN: llc -mtriple thumbv7k-apple-watchos %s -o - | FileCheck %s

define float @verify_aapcs_vfp(float %in) {
; CHECK: vadd.f32 s0, s0, s0

  %res = fadd float %in, %in
  ret float %res
}
