; RUN: not llc -mtriple=nvptx64 -filetype=null %s 2>&1 | FileCheck %s

; NVPTX has no fp128 soft-float library, so these conversions should diagnose a
; missing libcall rather than crash.

; CHECK: error: no libcall available for fp_extend
define fp128 @test_fpext_f64(double %x) nounwind {
  %r = fpext double %x to fp128
  ret fp128 %r
}

; CHECK: error: no libcall available for fp_round
define double @test_fptrunc_f128(fp128 %x) nounwind {
  %r = fptrunc fp128 %x to double
  ret double %r
}

; CHECK: error: no libcall available for fp_round
define half @test_fptrunc_f128_half(fp128 %x) nounwind {
  %r = fptrunc fp128 %x to half
  ret half %r
}

; CHECK: error: no libcall available for sint_to_fp
define fp128 @test_sitofp(i32 %x) nounwind {
  %r = sitofp i32 %x to fp128
  ret fp128 %r
}

; CHECK: error: no libcall available for uint_to_fp
define fp128 @test_uitofp(i32 %x) nounwind {
  %r = uitofp i32 %x to fp128
  ret fp128 %r
}

; CHECK: error: no libcall available for fp_to_sint
define i32 @test_fptosi(fp128 %x) nounwind {
  %r = fptosi fp128 %x to i32
  ret i32 %r
}

; CHECK: error: no libcall available for fp_to_uint
define i32 @test_fptoui(fp128 %x) nounwind {
  %r = fptoui fp128 %x to i32
  ret i32 %r
}

; CHECK: error: no libcall available for strict_fp_extend
define fp128 @test_strict_fpext(double %x) nounwind strictfp {
  %r = call fp128 @llvm.experimental.constrained.fpext.f128.f64(double %x, metadata !"fpexcept.strict")
  ret fp128 %r
}

; CHECK: error: no libcall available for strict_fp_round
define double @test_strict_fptrunc(fp128 %x) nounwind strictfp {
  %r = call double @llvm.experimental.constrained.fptrunc.f64.f128(fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret double %r
}

; CHECK: error: no libcall available for strict_sint_to_fp
define fp128 @test_strict_sitofp(i32 %x) nounwind strictfp {
  %r = call fp128 @llvm.experimental.constrained.sitofp.f128.i32(i32 %x, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret fp128 %r
}

; CHECK: error: no libcall available for strict_fp_to_sint
define i32 @test_strict_fptosi(fp128 %x) nounwind strictfp {
  %r = call i32 @llvm.experimental.constrained.fptosi.i32.f128(fp128 %x, metadata !"fpexcept.strict")
  ret i32 %r
}
