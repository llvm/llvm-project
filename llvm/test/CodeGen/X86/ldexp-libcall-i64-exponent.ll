; RUN: not llc -mtriple=x86_64-unknown-linux-gnu -filetype=null %s 2>&1 | FileCheck %s

; Check that llvm.ldexp with an exponent wider than the libcall's int
; fails cleanly rather than silently truncating.

; CHECK: error: ldexp exponent is wider than sizeof(int)
declare double @llvm.ldexp.f64.i64(double, i64)
define double @ldexp_f64_i64(double %x, i64 %e) {
  %r = call double @llvm.ldexp.f64.i64(double %x, i64 %e)
  ret double %r
}

; CHECK: error: ldexp exponent is wider than sizeof(int)
declare float @llvm.ldexp.f32.i64(float, i64)
define float @ldexp_f32_i64(float %x, i64 %e) {
  %r = call float @llvm.ldexp.f32.i64(float %x, i64 %e)
  ret float %r
}

; CHECK: error: ldexp exponent is wider than sizeof(int)
declare double @llvm.experimental.constrained.ldexp.f64.i64(double, i64, metadata, metadata)
define double @strict_ldexp_f64_i64(double %x, i64 %e) #0 {
  %r = call double @llvm.experimental.constrained.ldexp.f64.i64(double %x, i64 %e,
                                                                 metadata !"round.tonearest",
                                                                 metadata !"fpexcept.strict") #0
  ret double %r
}

attributes #0 = { strictfp }
