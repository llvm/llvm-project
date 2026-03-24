; RUN: not llc -mtriple=aarch64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; Verify that we get a user-friendly error instead of an assertion failure
; when trying to fpext to x86_fp80 on AArch64 (which has no libcall for it).
; See: https://github.com/llvm/llvm-project/issues/182449

; CHECK: error: do not know how to soften fp_extend

define x86_fp80 @test_fpext_double_to_x86_fp80(double %x) {
  %ext = fpext double %x to x86_fp80
  ret x86_fp80 %ext
}
