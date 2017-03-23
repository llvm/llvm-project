; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; This test is for a scheduler bug where VTX_READ instructions that used
; the result of another VTX_READ instruction were being grouped in the
; same fetch clasue.

; CHECK: {{^}}test:
; CHECK: Fetch clause
; CHECK: VTX_READ_32 [[IN0:T[0-9]+\.X]], [[IN0]], 0
; CHECK: Fetch clause
; CHECK: VTX_READ_32 [[IN1:T[0-9]+\.X]], [[IN1]], 0
define amdgpu_kernel void @test(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* addrspace(1)* nocapture %in0) {
entry:
  %0 = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(1)* %in0
  %1 = load i32, i32 addrspace(1)* %0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}
