; RUN: not --crash llc -mtriple=riscv32 -mattr=+v -global-isel -stop-after=irtranslator \
; RUN:   -verify-machineinstrs < %s 2>&1 | FileCheck %s
; RUN: not --crash llc -mtriple=riscv64 -mattr=+v -global-isel -stop-after=irtranslator \
; RUN:   -verify-machineinstrs < %s 2>&1 | FileCheck %s

; The purpose of this test is to show that the compiler throws an error when
; there is no support for bf16 vectors. If the compiler did not throw an error,
; then it will try to scalarize the argument to an s32, which may drop elements.
define void @test_args_nxv1bf16(<vscale x 1 x bfloat> %a) {
entry:
  ret void
}

; CHECK: LLVM ERROR: unable to lower arguments: void (<vscale x 1 x bfloat>) (in function: test_args_nxv1bf16)


