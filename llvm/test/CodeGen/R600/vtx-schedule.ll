; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; This test is for a scheduler bug where VTX_READ instructions that used
; the result of another VTX_READ instruction were being grouped in the
; same fetch clasue.

; CHECK: @test
; CHECK: Fetch clause
; CHECK_VTX_READ_32 [[IN0:T[0-9]+\.X]], [[IN0]], 40
; CHECK_VTX_READ_32 [[IN1:T[0-9]+\.X]], [[IN1]], 44
; CHECK: Fetch clause
; CHECK_VTX_READ_32 [[IN0:T[0-9]+\.X]], [[IN0]], 0
; CHECK_VTX_READ_32 [[IN1:T[0-9]+\.X]], [[IN1]], 0
define void @test(i32 addrspace(1)* nocapture %out, i32 addrspace(1)* nocapture %in0, i32 addrspace(1)* nocapture %in1) {
entry:
  %0 = load i32 addrspace(1)* %in0, align 4
  %1 = load i32 addrspace(1)* %in1, align 4
  %cmp.i = icmp slt i32 %0, %1
  %cond.i = select i1 %cmp.i, i32 %0, i32 %1
  store i32 %cond.i, i32 addrspace(1)* %out, align 4
  ret void
}
