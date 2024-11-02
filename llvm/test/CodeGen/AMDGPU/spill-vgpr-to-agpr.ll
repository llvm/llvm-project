; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX908 %s

; GFX908-LABEL: {{^}}max_11_vgprs_used_9a:
; GFX908-NOT: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GFX908-NOT: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-DAG: v_accvgpr_write_b32 [[A_REG:a[0-9]+]], v{{[0-9]}}
; GFX908-NOT: buffer_store_dword v{{[0-9]}},
; GFX908-NOT: buffer_
; GFX908:     v_mov_b32_e32 v{{[0-9]}}, [[V_REG:v[0-9]+]]
; GFX908:     v_accvgpr_read_b32 [[V_REG]], [[A_REG]]
; GFX908-NOT: buffer_

; GFX908: NumVgprs: 10
; GFX908: ScratchSize: 0
; GFX908: VGPRBlocks: 2
; GFX908: NumVGPRsForWavesPerEU: 10
define amdgpu_kernel void @max_11_vgprs_used_9a(i32 addrspace(1)* %p) #0 {
  %tid = load volatile i32, i32 addrspace(1)* undef
  call void asm sideeffect "", "a,a,a,a,a,a,a,a,a"(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9)
  %p1 = getelementptr inbounds i32, i32 addrspace(1)* %p, i32 %tid
  %p2 = getelementptr inbounds i32, i32 addrspace(1)* %p1, i32 4
  %p3 = getelementptr inbounds i32, i32 addrspace(1)* %p2, i32 8
  %p4 = getelementptr inbounds i32, i32 addrspace(1)* %p3, i32 12
  %p5 = getelementptr inbounds i32, i32 addrspace(1)* %p4, i32 16
  %p6 = getelementptr inbounds i32, i32 addrspace(1)* %p5, i32 20
  %p7 = getelementptr inbounds i32, i32 addrspace(1)* %p6, i32 24
  %p8 = getelementptr inbounds i32, i32 addrspace(1)* %p7, i32 28
  %p9 = getelementptr inbounds i32, i32 addrspace(1)* %p8, i32 32
  %p10 = getelementptr inbounds i32, i32 addrspace(1)* %p9, i32 36
  %v1 = load volatile i32, i32 addrspace(1)* %p1
  %v2 = load volatile i32, i32 addrspace(1)* %p2
  %v3 = load volatile i32, i32 addrspace(1)* %p3
  %v4 = load volatile i32, i32 addrspace(1)* %p4
  %v5 = load volatile i32, i32 addrspace(1)* %p5
  %v6 = load volatile i32, i32 addrspace(1)* %p6
  %v7 = load volatile i32, i32 addrspace(1)* %p7
  %v8 = load volatile i32, i32 addrspace(1)* %p8
  %v9 = load volatile i32, i32 addrspace(1)* %p9
  %v10 = load volatile i32, i32 addrspace(1)* %p10
  call void asm sideeffect "", "v,v,v,v,v,v,v,v,v,v"(i32 %v1, i32 %v2, i32 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10)
  store volatile i32 %v1, i32 addrspace(1)* undef
  store volatile i32 %v2, i32 addrspace(1)* undef
  store volatile i32 %v3, i32 addrspace(1)* undef
  store volatile i32 %v4, i32 addrspace(1)* undef
  store volatile i32 %v5, i32 addrspace(1)* undef
  store volatile i32 %v6, i32 addrspace(1)* undef
  store volatile i32 %v7, i32 addrspace(1)* undef
  store volatile i32 %v8, i32 addrspace(1)* undef
  store volatile i32 %v9, i32 addrspace(1)* undef
  store volatile i32 %v10, i32 addrspace(1)* undef
  ret void
}

; GFX908-LABEL: {{^}}max_11_vgprs_used_1a_partial_spill:
; GFX908-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GFX908-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-DAG: v_accvgpr_write_b32 a0, 1
; GFX908-DAG:    buffer_store_dword v{{[0-9]}},
; GFX908-DAG: v_accvgpr_write_b32 a1, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a2, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a3, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a4, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a5, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a6, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a7, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a8, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a9, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a10, v{{[0-9]}}
; GFX908-DAG:    buffer_load_dword v{{[0-9]}},
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a0
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a1
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a2
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a3
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a4
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a5
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a6
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a7
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a8
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a9
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a10

; GFX908: NumVgprs: 10
; GFX908: ScratchSize: 12
; GFX908: VGPRBlocks: 2
; GFX908: NumVGPRsForWavesPerEU: 11
define amdgpu_kernel void @max_11_vgprs_used_1a_partial_spill(i64 addrspace(1)* %p) #0 {
  %tid = load volatile i32, i32 addrspace(1)* undef
  call void asm sideeffect "", "a"(i32 1)
  %p1 = getelementptr inbounds i64, i64 addrspace(1)* %p, i32 %tid
  %p2 = getelementptr inbounds i64, i64 addrspace(1)* %p1, i32 8
  %p3 = getelementptr inbounds i64, i64 addrspace(1)* %p2, i32 16
  %p4 = getelementptr inbounds i64, i64 addrspace(1)* %p3, i32 24
  %p5 = getelementptr inbounds i64, i64 addrspace(1)* %p4, i32 32
  %v1 = load volatile i64, i64 addrspace(1)* %p1
  %v2 = load volatile i64, i64 addrspace(1)* %p2
  %v3 = load volatile i64, i64 addrspace(1)* %p3
  %v4 = load volatile i64, i64 addrspace(1)* %p4
  %v5 = load volatile i64, i64 addrspace(1)* %p5
  call void asm sideeffect "", "v,v,v,v,v"(i64 %v1, i64 %v2, i64 %v3, i64 %v4, i64 %v5)
  store volatile i64 %v1, i64 addrspace(1)* %p2
  store volatile i64 %v2, i64 addrspace(1)* %p3
  store volatile i64 %v3, i64 addrspace(1)* %p4
  store volatile i64 %v4, i64 addrspace(1)* %p5
  store volatile i64 %v5, i64 addrspace(1)* %p1
  ret void
}

attributes #0 = { nounwind "amdgpu-num-vgpr"="11" }
