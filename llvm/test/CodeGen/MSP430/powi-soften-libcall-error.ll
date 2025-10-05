; RUN: not llc -mtriple=msp430 -filetype=null %s 2>&1 | FileCheck %s

; FIXME: This should not fail but isn't implemented
; CHECK: error: powi exponent does not match sizeof(int)
define float @soften_powi_error(float %x, i32 %n) {
  %powi = call float @llvm.powi.f32.i32(float %x, i32 %n)
  ret float %powi
}

; CHECK: error: powi exponent does not match sizeof(int)
define float @soften_powi_error_strictfp(float %x, i32 %n) strictfp {
 %powi = call float @llvm.experimental.constrained.powi.f32.i32(float %x, i32 %n, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret float %powi
}


