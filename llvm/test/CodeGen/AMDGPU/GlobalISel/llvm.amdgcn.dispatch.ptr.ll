; RUN: llc -global-isel -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: Error on non-HSA target

; GCN-LABEL: {{^}}test:
; GCN: enable_sgpr_dispatch_ptr = 1
; GCN: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
define amdgpu_kernel void @test(i32 addrspace(1)* %out) {
  %dispatch_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
  %header_ptr = bitcast i8 addrspace(4)* %dispatch_ptr to i32 addrspace(4)*
  %value = load i32, i32 addrspace(4)* %header_ptr
  store i32 %value, i32 addrspace(1)* %out
  ret void
}

declare noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0

attributes #0 = { readnone }
