; RUN: llc -mtriple=thumbv8m.main-none-eabi %s -o - | FileCheck %s

; Check that freeze operations on floating types are successfully softened.

; CHECK-LABEL: sitofp_f32_i32:
; CHECK: bl __aeabi_i2f
define float @sitofp_f32_i32(i32 %x) #0 {
  %val = call float @llvm.experimental.constrained.sitofp.f32.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict") #0
  %val.fr = freeze float %val
  ret float %val.fr
}

attributes #0 = { strictfp }

declare float @llvm.experimental.constrained.sitofp.f32.i32(i32, metadata, metadata)
declare double @llvm.experimental.constrained.sitofp.f64.i32(i32, metadata, metadata)
