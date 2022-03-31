; RUN: not opt -verify < %s 2>&1 | FileCheck %s

declare half @llvm.fptrunc.round(float, metadata)

define void @test_fptrunc_round_dynamic(float %a) {
; CHECK: unsupported rounding mode argument
  %res = call half @llvm.fptrunc.round(float %a, metadata !"round.dynamic")
; CHECK: unsupported rounding mode argument
  %res1 = call half @llvm.fptrunc.round(float %a, metadata !"round.test")
; CHECK: invalid value for llvm.fptrunc.round metadata operand (the operand should be a string)
  %res2 = call half @llvm.fptrunc.round(float %a, metadata i32 5)
  ret void
}
