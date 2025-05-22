; RUN: not --crash llc -mtriple=riscv32 -mattr=+v -global-isel -stop-after=irtranslator \
; RUN:   -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=riscv64 -mattr=+v -global-isel -stop-after=irtranslator \
; RUN:   -verify-machineinstrs < %s 2>&1 | FileCheck %s

; The purpose of this test is to show that the compiler throws an error when
; there is no support for bf16 vectors. If the compiler did not throw an error,
; then it will try to scalarize the argument to an s32, which may drop elements.
define <vscale x 1 x bfloat> @test_ret_nxv1bf16() {
entry:
  ret <vscale x 1 x bfloat> undef
}

; CHECK: LLVM ERROR: unable to translate instruction: ret (in function: test_ret_nxv1bf16)
