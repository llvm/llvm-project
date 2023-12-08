; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; This test checks a bug in R600TargetLowering::LowerSELECT_CC where the
; chance to optimize the fcmp + select instructions to SET* was missed
; due to the fact that the operands to fcmp and select had different types

; CHECK: SET{{[A-Z]+}}_DX10

define amdgpu_kernel void @test(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %0 = load float, ptr addrspace(1) %in
  %cmp = fcmp oeq float %0, 0.000000e+00
  %value = select i1 %cmp, i32 -1, i32 0
  store i32 %value, ptr addrspace(1) %out
  ret void
}
