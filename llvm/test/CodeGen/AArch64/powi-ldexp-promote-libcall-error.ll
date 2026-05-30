; RUN: not llc -mtriple=aarch64-linux-gnu -filetype=null %s 2>&1 | FileCheck %s

; A powi/ldexp exponent wider than sizeof(int) can't be passed to the libcall,
; so PromoteIntOp_ExpOp must emit a clean error rather than crash (assertions)
; or silently truncate. i48 is illegal and promoted through that path here; i64
; would be legal and i128 expanded, so neither would reach it.

; CHECK: error: powi/ldexp exponent does not match sizeof(int)
define double @ldexp_f64_i48(double %val, i48 %a) {
  %call = call double @llvm.ldexp.f64.i48(double %val, i48 %a)
  ret double %call
}

; CHECK: error: powi/ldexp exponent does not match sizeof(int)
define double @powi_f64_i48(double %val, i48 %a) {
  %call = call double @llvm.powi.f64.i48(double %val, i48 %a)
  ret double %call
}

; CHECK: error: powi/ldexp exponent does not match sizeof(int)
define double @ldexp_f64_i48_strictfp(double %val, i48 %a) strictfp {
  %call = call double @llvm.experimental.constrained.ldexp.f64.i48(double %val, i48 %a, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %call
}
