; RUN: llc -fast-isel -fast-isel-abort=1 -mtriple=x86_64-unknown-unknown -mattr=+f16c < %s

; XFAIL: *

; In the future, we might want to teach fast-isel how to expand a double-to-half
; conversion into a double-to-float conversion immediately followed by a
; float-to-half conversion. For now, fast-isel is expected to fail.

define double @test_fp16_to_fp64(i32 %a) {
entry:
  %0 = trunc i32 %a to i16
  %1 = call double @llvm.convert.from.fp16.f64(i16 %0)
  ret float %0
}

define i16 @test_fp64_to_fp16(double %a) {
entry:
  %0 = call i16 @llvm.convert.to.fp16.f64(double %a)
  ret i16 %0
}

declare i16 @llvm.convert.to.fp16.f64(double)
declare double @llvm.convert.from.fp16.f64(i16)
