; RUN: llc -global-isel -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: Error on non-hsa target

; GCN-LABEL: {{^}}test:
; GCN: enable_sgpr_queue_ptr = 1
; GCN: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
define amdgpu_kernel void @test(ptr addrspace(1) %out) {
  %queue_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %value = load i32, ptr addrspace(4) %queue_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

declare noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0

attributes #0 = { nounwind readnone }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 200}
