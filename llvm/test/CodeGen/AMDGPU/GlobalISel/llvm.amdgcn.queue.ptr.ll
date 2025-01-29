; RUN: llc -global-isel -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: Error on non-hsa target

; GCN-LABEL: {{^}}queue_ptr:
; GCN: s_load_dword s{{[0-9]+}}, s[6:7], 0x0
; GCN: .amdhsa_user_sgpr_queue_ptr 1
define amdgpu_kernel void @queue_ptr(ptr addrspace(1) %out) {
  %queue_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %value = load i32, ptr addrspace(4) %queue_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}queue_ptr_opt:
; GCN: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; GCN: .amdhsa_user_sgpr_queue_ptr 1
define amdgpu_kernel void @queue_ptr_opt(ptr addrspace(1) %out) #1 {
  %queue_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %value = load i32, ptr addrspace(4) %queue_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

declare noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0

attributes #0 = { nounwind readnone }
attributes #1 = { "amdgpu-no-dispatch-ptr" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
