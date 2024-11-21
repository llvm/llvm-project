; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: not llc -mtriple=amdgcn-unknown-unknown -mcpu=kaveri -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: in function test{{.*}}: unsupported hsa intrinsic without hsa target

; GCN-LABEL: {{^}}test:
; GCN: s_load_dword s{{[0-9]+}}, s[6:7], 0x0
; GCN: .amdhsa_user_sgpr_queue_ptr 1
define amdgpu_kernel void @test(ptr addrspace(1) %out) {
  %queue_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %value = load i32, ptr addrspace(4) %queue_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; FIXME: Should really be able to delete the load
; GCN-LABEL: {{^}}test_ub:
; GCN: s_load_dword s{{[0-9]+}}, s[0:1], 0x0
; GCN: .amdhsa_user_sgpr_queue_ptr 0
define amdgpu_kernel void @test_ub(ptr addrspace(1) %out) #1 {
  %queue_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0
  %value = load i32, ptr addrspace(4) %queue_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

declare noalias ptr addrspace(4) @llvm.amdgcn.queue.ptr() #0

attributes #0 = { nounwind readnone }
attributes #1 = { "amdgpu-no-queue-ptr" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdhsa_code_object_version", i32 400}
