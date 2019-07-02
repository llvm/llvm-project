; RUN: llc -global-isel -mtriple=amdgcn-mesa-mesa3d -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: Dropped parts from original test

; GCN-LABEL: {{^}}test_ps:
; GCN: s_load_dword s{{[0-9]+}}, s[0:1], 0x0
define amdgpu_ps i32 @test_ps() #1 {
  %implicit_buffer_ptr = call i8 addrspace(4)* @llvm.amdgcn.implicit.buffer.ptr()
  %buffer_ptr = bitcast i8 addrspace(4)* %implicit_buffer_ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %buffer_ptr
  ret i32 %value
}

declare i8 addrspace(4)* @llvm.amdgcn.implicit.buffer.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }
