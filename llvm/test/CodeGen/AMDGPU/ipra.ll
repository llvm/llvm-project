; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -enable-ipra < %s | FileCheck -check-prefix=GCN %s

; Kernels are not called, so there is no call preserved mask.
; GCN-LABEL: {{^}}kernel:
; GCN: flat_store_dword
define amdgpu_kernel void @kernel(i32 addrspace(1)* %out) #0 {
entry:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
