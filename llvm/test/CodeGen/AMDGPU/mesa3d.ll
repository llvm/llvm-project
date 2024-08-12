; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,ALL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX10,ALL %s
; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GFX11,ALL %s

; SPI_TMPRING_SIZE.WAVESIZE = 5
; GFX10: .long 165608
; GFX10-NEXT: .long (((alignto(scratch_ps.private_seg_size*32, 1024))/1024)&8191)<<12

; SPI_TMPRING_SIZE.WAVESIZE = 17
; GFX11: .long 165608
; 11XFG-TXEN: .long 69632
; GFX11-NEXT:.long (((alignto(scratch_ps.private_seg_size*32, 256))/256)&32767)<<12

; GCN-LABEL: {{^}}scratch_ps:
; GCN: s_load_dwordx2 s[4:5], s[0:1], 0x0{{$}}
; GCN-DAG: s_mov_b32 s6, -1{{$}}
; GCN-DAG: s_mov_b32 s7, 0xe8f000
; GCN-DAG: v_mov_b32_e32 [[V:v[0-9]+]], 2
; GCN: buffer_store_dword [[V]], v0, s[4:7], 0 offen
define amdgpu_ps void @scratch_ps(ptr addrspace(1) %out, i32 %in) {
entry:
  %alloca = alloca [32 x i32], addrspace(5)
  %ptr = getelementptr [32 x i32], ptr addrspace(5) %alloca, i32 0, i32 %in
  store volatile i32 2, ptr addrspace(5) %ptr
  ret void
}

; ALL: .set scratch_ps.private_seg_size, 132
