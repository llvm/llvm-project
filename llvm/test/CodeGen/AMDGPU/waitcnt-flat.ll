; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=kaveri | FileCheck --check-prefix=GCN %s
; RUN: llc < %s -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global | FileCheck --check-prefix=GCN %s

; If flat_store_dword and flat_load_dword use different registers for the data
; operand, this test is not broken.  It just means it is no longer testing
; for the original bug.

; GCN: {{^}}test:
; XGCN: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[DATA:v[0-9]+]]
; XGCN: s_waitcnt vmcnt(0) lgkmcnt(0)
; XGCN: flat_load_dword [[DATA]], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test(i32 addrspace(1)* %out, i32 %in) {
  store volatile i32 0, i32 addrspace(1)* %out
  %val = load volatile i32, i32 addrspace(1)* %out
  ret void
}
