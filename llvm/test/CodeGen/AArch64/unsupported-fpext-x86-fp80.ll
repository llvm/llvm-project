; RUN: not llc -mtriple=aarch64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; Verify that we get a user-friendly error instead of an assertion failure
; when trying to fpext to x86_fp80 on AArch64 (which has no libcall for it).
; See: https://github.com/llvm/llvm-project/issues/182449
;
; Targets that support the x87 format define the `FPEXT_*_F80` libcalls, so the
; pair above is known and the error comes from the missing implementation
; rather than from the missing libcall.

; CHECK: error: no libcall available for fp_extend with type f80

define x86_fp80 @test_fpext_double_to_x86_fp80(double %x) {
  %ext = fpext double %x to x86_fp80
  ret x86_fp80 %ext
}
