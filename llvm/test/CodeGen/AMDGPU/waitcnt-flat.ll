; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -mattr=-flat-for-global < %s | FileCheck -enable-var-scope -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX9 %s

; If flat_store_dword and flat_load_dword use different registers for the data
; operand, this test is not broken.  It just means it is no longer testing
; for the original bug.

; GCN: {{^}}test:
; XGCN: flat_store_dword v[{{[0-9]+:[0-9]+}}], [[DATA:v[0-9]+]]
; XGCN: s_waitcnt vmcnt(0) lgkmcnt(0)
; XGCN: flat_load_dword [[DATA]], v[{{[0-9]+:[0-9]+}}]
define amdgpu_kernel void @test(i32 addrspace(4)* %out, i32 %in) {
  store volatile i32 0, i32 addrspace(4)* %out
  %val = load volatile i32, i32 addrspace(4)* %out
  ret void
}

; Make sure lgkmcnt isn't used for global_* instructions
; GCN-LABEL: {{^}}test_waitcnt_type_flat_global:
; GFX9: global_load_dword [[LD:v[0-9]+]]
; GFX9-NEXT: s_waitcnt vmcnt(0){{$}}
; GFX9-NEXT: ds_write_b32 [[LD]]
define amdgpu_kernel void @test_waitcnt_type_flat_global(i32 addrspace(1)* %in) {
  %val = load volatile i32, i32 addrspace(1)* %in
  store volatile i32 %val, i32 addrspace(3)* undef
  ret void
}
