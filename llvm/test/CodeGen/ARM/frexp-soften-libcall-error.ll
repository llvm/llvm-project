; RUN: not llc -mtriple arm-linux-gnu -float-abi=soft -filetype=null %s 2>&1 | FileCheck %s

; FIXME: This should not fail but isn't implemented
; CHECK: error: ffrexp exponent does not match sizeof(int)
define { float, i64 } @soften_frexp_error(float %x) {
  %frexp = call { float, i64 } @llvm.frexp.f32.i64(float %x)
  ret { float, i64 } %frexp
}
