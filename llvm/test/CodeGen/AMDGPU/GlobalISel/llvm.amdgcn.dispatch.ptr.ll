; RUN: llc -global-isel -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: Error on non-HSA target

; GCN-LABEL: {{^}}test:
; GCN: enable_sgpr_dispatch_ptr = 1
; GCN: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
define amdgpu_kernel void @test(ptr addrspace(1) %out) {
  %dispatch_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
  %value = load i32, ptr addrspace(4) %dispatch_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

declare noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0

attributes #0 = { readnone }
