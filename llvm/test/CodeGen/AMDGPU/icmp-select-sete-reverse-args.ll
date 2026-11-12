;RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s

;Test that a select with reversed True/False values is correctly lowered
;to a SETNE_INT.  There should only be one SETNE_INT instruction.

;CHECK: SETNE_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
;CHECK-NOT: SETNE_INT

define amdgpu_kernel void @test(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %0 = load i32, ptr addrspace(1) %in
  %arrayidx1 = getelementptr inbounds i32, ptr addrspace(1) %in, i32 1
  %1 = load i32, ptr addrspace(1) %arrayidx1
  %cmp = icmp eq i32 %0, %1
  %value = select i1 %cmp, i32 0, i32 -1
  store i32 %value, ptr addrspace(1) %out
  ret void
}
