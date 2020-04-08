; RUN: llc -mtriple=amdgcn-mesa-mesa3d -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: Requires stack object to not assert
; GCN-LABEL: {{^}}test_ps:
; GCN: s_load_dwordx2 s[4:5], s[0:1], 0x0
; GCN: buffer_store_dword v0, off, s[4:7], s2 offset:4
; GCN: s_load_dword s{{[0-9]+}}, s[0:1], 0x0
; GCN-NEXT: s_waitcnt
; GCN-NEXT: ; return
define amdgpu_ps i32 @test_ps() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  %implicit_buffer_ptr = call i8 addrspace(4)* @llvm.amdgcn.implicit.buffer.ptr()
  %buffer_ptr = bitcast i8 addrspace(4)* %implicit_buffer_ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %buffer_ptr
  ret i32 %value
}

; GCN-LABEL: {{^}}test_cs:
; GCN: s_mov_b64 s[4:5], s[0:1]
; GCN: buffer_store_dword v{{[0-9]+}}, off, s[4:7], s2 offset:4
; GCN: s_load_dword s0, s[0:1], 0x0
define amdgpu_cs i32 @test_cs() #1 {
  %alloca = alloca i32, addrspace(5)
  store volatile i32 0, i32 addrspace(5)* %alloca
  %implicit_buffer_ptr = call i8 addrspace(4)* @llvm.amdgcn.implicit.buffer.ptr()
  %buffer_ptr = bitcast i8 addrspace(4)* %implicit_buffer_ptr to i32 addrspace(4)*
  %value = load volatile i32, i32 addrspace(4)* %buffer_ptr
  ret i32 %value
}

declare i8 addrspace(4)* @llvm.amdgcn.implicit.buffer.ptr() #0

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind }
