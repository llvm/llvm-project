; RUN: llc -march=amdgcn -mcpu=gfx1010 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX10 %s

declare i32 @llvm.amdgcn.s.get.waveid.in.workgroup() #0

; GCN-LABEL: {{^}}test_s_get_waveid_in_workgroup:
; GFX10: global_store_dword
; GFX10: s_get_waveid_in_workgroup [[DEST:s[0-9]+]]
; GFX10: s_waitcnt lgkmcnt(0)
; GFX10: v_mov_b32_e32 [[VDEST:v[0-9]+]], [[DEST]]
; GFX10: global_store_dword v{{[0-9]+}}, [[VDEST]], s{{\[[0-9]+:[0-9]+\]$}}
define amdgpu_kernel void @test_s_get_waveid_in_workgroup(ptr addrspace(1) %out) {
; Make sure %out is loaded and assiciated wait count already inserted
  store i32 0, ptr addrspace(1) %out
  %v = call i32 @llvm.amdgcn.s.get.waveid.in.workgroup()
  store i32 %v, ptr addrspace(1) %out
  ret void
}

attributes #0 = { nounwind }
