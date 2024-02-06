; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: not llc -mtriple=amdgcn-unknown-unknown -mcpu=kaveri -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: in function test{{.*}}: unsupported hsa intrinsic without hsa target

; GCN-LABEL: {{^}}test:
; GCN: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; GCN: .amdhsa_user_sgpr_dispatch_ptr 1
define amdgpu_kernel void @test(ptr addrspace(1) %out) {
  %dispatch_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
  %value = load i32, ptr addrspace(4) %dispatch_ptr
  store i32 %value, ptr addrspace(1) %out
  ret void
}

; GCN-LABEL: {{^}}test2
; GCN: s_load_dword s[[REG:[0-9]+]], s[4:5], 0x1
; GCN: s_lshr_b32 s{{[0-9]+}}, s[[REG]], 16
; GCN-NOT: load_ushort
; GCN: s_endpgm
; GCN: .amdhsa_user_sgpr_dispatch_ptr 1
define amdgpu_kernel void @test2(ptr addrspace(1) %out) {
  %dispatch_ptr = call noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0
  %d1 = getelementptr inbounds i8, ptr addrspace(4) %dispatch_ptr, i64 6
  %v1 = load i16, ptr addrspace(4) %d1
  %e1 = zext i16 %v1 to i32
  store i32 %e1, ptr addrspace(1) %out
  ret void
}

declare noalias ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #0

attributes #0 = { readnone }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 400}
