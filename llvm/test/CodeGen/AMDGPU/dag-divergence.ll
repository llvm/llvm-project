; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}private_load_maybe_divergent:
; GCN: buffer_load_dword
; GCN-NOT: s_load_dword s
; GCN: flat_load_dword
; GCN-NOT: s_load_dword s
define amdgpu_kernel void @private_load_maybe_divergent(i32 addrspace(4)* %k, i32* %flat) {
  %load = load volatile i32, i32 addrspace(5)* undef, align 4
  %gep = getelementptr inbounds i32, i32 addrspace(4)* %k, i32 %load
  %maybe.not.uniform.load = load i32, i32 addrspace(4)* %gep, align 4
  store i32 %maybe.not.uniform.load, i32 addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}flat_load_maybe_divergent:
; GCN: s_load_dwordx4
; GCN-NOT: s_load
; GCN: flat_load_dword
; GCN-NOT: s_load
; GCN: flat_load_dword
; GCN-NOT: s_load
; GCN: flat_store_dword
define amdgpu_kernel void @flat_load_maybe_divergent(i32 addrspace(4)* %k, i32* %flat) {
  %load = load i32, i32* %flat, align 4
  %gep = getelementptr inbounds i32, i32 addrspace(4)* %k, i32 %load
  %maybe.not.uniform.load = load i32, i32 addrspace(4)* %gep, align 4
  store i32 %maybe.not.uniform.load, i32 addrspace(1)* undef
  ret void
}
