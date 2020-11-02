; RUN: llc -mtriple=amdgcn--amdhsa --amdhsa-code-object-version=2 -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: not llc -mtriple=amdgcn-unknown-unknown -mcpu=kaveri -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERROR %s

; ERROR: in function test{{.*}}: unsupported hsa intrinsic without hsa target

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

; GCN-LABEL: {{^}}test2
; GCN: enable_sgpr_dispatch_ptr = 1
; GCN: s_load_dword s[[REG:[0-9]+]], s[4:5], 0x1
; GCN: s_lshr_b32 s{{[0-9]+}}, s[[REG]], 16
; GCN-NOT: load_ushort
; GCN: s_endpgm
define amdgpu_kernel void @test2(i32 addrspace(1)* %out) {
  %dispatch_ptr = call noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0
  %d1 = getelementptr inbounds i8, i8 addrspace(4)* %dispatch_ptr, i64 6
  %h1 = bitcast i8 addrspace(4)* %d1 to i16 addrspace(4)*
  %v1 = load i16, i16 addrspace(4)* %h1
  %e1 = zext i16 %v1 to i32
  store i32 %e1, i32 addrspace(1)* %out
  ret void
}

declare noalias i8 addrspace(4)* @llvm.amdgcn.dispatch.ptr() #0

attributes #0 = { readnone }
