; RUN: not llc -mtriple=arm-linux-gnu -float-abi=soft -filetype=null %s 2>&1 | FileCheck %s

; FIXME: This should not fail but isn't implemented
; CHECK: error: powi exponent does not match sizeof(int)
define float @soften_powi_error(float %x, i64 %n) {
  %powi = call float @llvm.powi.f32.i64(float %x, i64 %n)
  ret float %powi
}
