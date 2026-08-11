; RUN: not llc -mtriple=i686-unknown-unknown -filetype=null %s 2>&1 | FileCheck %s

; Make sure these missing libcalls emit a proper diagnostic rather
; than fatal erroring.

; CHECK: error: no libcall available for llrint
define i64 @test_llrint_ppcf128(ppc_fp128 %x) nounwind {
  %r = call i64 @llvm.llrint.i64.ppcf128(ppc_fp128 %x)
  ret i64 %r
}

; CHECK: error: no libcall available for llround
define i64 @test_llround_ppcf128(ppc_fp128 %x) nounwind {
  %r = call i64 @llvm.llround.i64.ppcf128(ppc_fp128 %x)
  ret i64 %r
}

; CHECK: error: no libcall available for strict_llrint
define i64 @test_strict_llrint_ppcf128(ppc_fp128 %x) nounwind strictfp {
  %r = call i64 @llvm.experimental.constrained.llrint.i64.ppcf128(ppc_fp128 %x, metadata !"round.tonearest", metadata !"fpexcept.strict")
  ret i64 %r
}

; CHECK: error: no libcall available for strict_llround
define i64 @test_strict_llround_ppcf128(ppc_fp128 %x) nounwind strictfp {
  %r = call i64 @llvm.experimental.constrained.llround.i64.ppcf128(ppc_fp128 %x, metadata !"fpexcept.strict")
  ret i64 %r
}
