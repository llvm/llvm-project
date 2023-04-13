; RUN: not --crash llc -march=amdgcn -mcpu=gfx1200 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=GFX12 %s

; GDS is not supported on GFX12+
; GFX12: LLVM ERROR: Cannot select: {{.*}} AtomicLoadAdd

define amdgpu_kernel void @atomic_add_ret_gds(ptr addrspace(1) %out, ptr addrspace(2) %gds) #1 {
  %val = atomicrmw volatile add ptr addrspace(2) %gds, i32 5 acq_rel
  store i32 %val, ptr addrspace(1) %out
  ret void
}
