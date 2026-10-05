; RUN: not llc -mtriple=x86_64-unknown-unknown -filetype=null %s 2>&1 | FileCheck %s

; x86 expands ppc_fp128 but has no ppc_fp128 runtime library, so these
; operations should diagnose a missing libcall rather than crash.

; CHECK: error: no libcall available for fsqrt with type ppcf128
define ppc_fp128 @test_sqrt(ppc_fp128 %x) nounwind {
  %r = call ppc_fp128 @llvm.sqrt.ppcf128(ppc_fp128 %x)
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for fpow with type ppcf128
define ppc_fp128 @test_pow(ppc_fp128 %x, ppc_fp128 %y) nounwind {
  %r = call ppc_fp128 @llvm.pow.ppcf128(ppc_fp128 %x, ppc_fp128 %y)
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for fma with type ppcf128
define ppc_fp128 @test_fma(ppc_fp128 %x, ppc_fp128 %y, ppc_fp128 %z) nounwind {
  %r = call ppc_fp128 @llvm.fma.ppcf128(ppc_fp128 %x, ppc_fp128 %y, ppc_fp128 %z)
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for fldexp with type ppcf128
define ppc_fp128 @test_ldexp(ppc_fp128 %x, i32 %y) nounwind {
  %r = call ppc_fp128 @llvm.ldexp.ppcf128.i32(ppc_fp128 %x, i32 %y)
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for fpowi with type ppcf128
define ppc_fp128 @test_powi(ppc_fp128 %x, i32 %y) nounwind {
  %r = call ppc_fp128 @llvm.powi.ppcf128.i32(ppc_fp128 %x, i32 %y)
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for strict_fldexp with type ppcf128
define ppc_fp128 @test_strict_ldexp(ppc_fp128 %x, i32 %y) nounwind strictfp {
  %r = call ppc_fp128 @llvm.experimental.constrained.ldexp.ppcf128.i32(ppc_fp128 %x, i32 %y, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for frem with type ppcf128
define ppc_fp128 @test_frem(ppc_fp128 %x, ppc_fp128 %y) nounwind {
  %r = frem ppc_fp128 %x, %y
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for sint_to_fp with type ppcf128
define ppc_fp128 @test_sitofp(i64 %x) nounwind {
  %r = sitofp i64 %x to ppc_fp128
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for fp_to_sint with type ppcf128
define i64 @test_fptosi(ppc_fp128 %x) nounwind {
  %r = fptosi ppc_fp128 %x to i64
  ret i64 %r
}

; CHECK: error: no libcall available for fsincos with type ppcf128
define { ppc_fp128, ppc_fp128 } @test_sincos(ppc_fp128 %x) nounwind {
  %r = call { ppc_fp128, ppc_fp128 } @llvm.sincos.ppcf128(ppc_fp128 %x)
  ret { ppc_fp128, ppc_fp128 } %r
}

; CHECK: error: no libcall available for fmodf with type ppcf128
define { ppc_fp128, ppc_fp128 } @test_modf(ppc_fp128 %x) nounwind {
  %r = call { ppc_fp128, ppc_fp128 } @llvm.modf.ppcf128(ppc_fp128 %x)
  ret { ppc_fp128, ppc_fp128 } %r
}

; CHECK: error: no libcall available for strict_fsqrt with type ppcf128
define ppc_fp128 @test_strict_sqrt(ppc_fp128 %x) nounwind strictfp {
  %r = call ppc_fp128 @llvm.experimental.constrained.sqrt.ppcf128(ppc_fp128 %x, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for strict_fadd with type ppcf128
define ppc_fp128 @test_strict_fadd(ppc_fp128 %x, ppc_fp128 %y) nounwind strictfp {
  %r = call ppc_fp128 @llvm.experimental.constrained.fadd.ppcf128(ppc_fp128 %x, ppc_fp128 %y, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for strict_fma with type ppcf128
define ppc_fp128 @test_strict_fma(ppc_fp128 %x, ppc_fp128 %y, ppc_fp128 %z) nounwind strictfp {
  %r = call ppc_fp128 @llvm.experimental.constrained.fma.ppcf128(ppc_fp128 %x, ppc_fp128 %y, ppc_fp128 %z, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for strict_sint_to_fp with type ppcf128
define ppc_fp128 @test_strict_sitofp(i64 %x) nounwind strictfp {
  %r = call ppc_fp128 @llvm.experimental.constrained.sitofp.ppcf128.i64(i64 %x, metadata !"round.dynamic", metadata !"fpexcept.strict")
  ret ppc_fp128 %r
}

; CHECK: error: no libcall available for strict_fp_to_sint with type ppcf128
define i64 @test_strict_fptosi(ppc_fp128 %x) nounwind strictfp {
  %r = call i64 @llvm.experimental.constrained.fptosi.i64.ppcf128(ppc_fp128 %x, metadata !"fpexcept.strict")
  ret i64 %r
}
