; RUN: llc -mtriple=amdgpu9.00 -filetype=obj < %s | llvm-objdump --triple=amdgpu9.00 --disassemble - | FileCheck %s

; Make sure the computed instruction size for LDS-DMA buffer loads is correct
; and passes the instruction size verifier. The offset/cpol/swz fields are
; packed into the instruction word and must not be counted as a trailing
; literal, so each load is 8 bytes (two 32-bit words).

declare void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8), ptr addrspace(3) nocapture, i32, i32, i32, i32, i32)

; CHECK: buffer_load_dword v0, s[0:3], 0 offen lds{{.*}}E0511000 80000000
define amdgpu_ps void @buffer_load_lds_dword_offen(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 2048, i32 0, i32 0, i32 0)
  ret void
}

; CHECK: buffer_load_dword off, s[0:3], 0 lds{{.*}}E0510000 80000000
define amdgpu_ps void @buffer_load_lds_dword_offset(ptr addrspace(8) inreg %rsrc, ptr addrspace(3) inreg %lds) {
  call void @llvm.amdgcn.raw.ptr.buffer.load.lds(ptr addrspace(8) %rsrc, ptr addrspace(3) %lds, i32 4, i32 0, i32 0, i32 0, i32 0)
  ret void
}
