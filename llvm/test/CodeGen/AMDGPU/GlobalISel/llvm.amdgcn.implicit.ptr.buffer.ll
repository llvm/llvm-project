; RUN: llc -global-isel -new-reg-bank-select -mtriple=amdgcn-mesa-mesa3d < %s | FileCheck -check-prefix=GCN %s

; FIXME: Dropped parts from original test

; GCN-LABEL: {{^}}test_ps:
; GCN: s_load_dword s{{[0-9]+}}, s[0:1], 0x0
define amdgpu_ps i32 @test_ps() #1 {
  %implicit_buffer_ptr = call ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr()
  %value = load volatile i32, ptr addrspace(4) %implicit_buffer_ptr
  ret i32 %value
}

declare ptr addrspace(4) @llvm.amdgcn.implicit.buffer.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }
