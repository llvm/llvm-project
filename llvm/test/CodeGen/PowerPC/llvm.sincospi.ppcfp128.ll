; XFAIL: *
; UNSUPPORTED: expensive_checks
; FIXME: asserts
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-gnu-linux -filetype=null -enable-legalize-types-checking=0 \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names %s

define { ppc_fp128, ppc_fp128 } @test_sincospi_ppcf128(ppc_fp128 %a) {
  %result = call { ppc_fp128, ppc_fp128 } @llvm.sincospi.ppcf128(ppc_fp128 %a)
  ret { ppc_fp128, ppc_fp128 } %result
}

; FIXME: This could be made a tail call with the default expansion of llvm.sincospi.
define void @test_sincospi_ppcf128_void_tail_call(ppc_fp128 %a, ptr noalias %out_sin, ptr noalias %out_cos) {
  %result = tail call { ppc_fp128, ppc_fp128 } @llvm.sincospi.ppcf128(ppc_fp128 %a)
  %result.0 = extractvalue { ppc_fp128, ppc_fp128 } %result, 0
  %result.1 = extractvalue { ppc_fp128, ppc_fp128 } %result, 1
  store ppc_fp128 %result.0, ptr %out_sin, align 16
  store ppc_fp128 %result.1, ptr %out_cos, align 16
  ret void
}

; NOTE: This would need a struct-return library call for llvm.sincospi to become a tail call.
define { ppc_fp128, ppc_fp128 } @test_sincospi_ppcf128_tail_call(ppc_fp128 %a) {
  %result = tail call { ppc_fp128, ppc_fp128 } @llvm.sincospi.ppcf128(ppc_fp128 %a)
  ret { ppc_fp128, ppc_fp128 } %result
}
