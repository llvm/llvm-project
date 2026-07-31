; RUN: llc -mtriple=amdgpu9.08 -amdgpu-stress-vgpr=11 -amdgpu-stress-agpr=11 < %s | FileCheck -check-prefixes=GCN,GFX908 %s
; RUN: llc -mtriple=amdgpu9.00 -amdgpu-stress-vgpr=11 < %s | FileCheck -check-prefixes=GCN,GFX900 %s

; GCN-LABEL: {{^}}max_11_vgprs:
; GFX900-NOT: SCRATCH_RSRC
; GFX908-NOT: SCRATCH_RSRC
; GFX908-DAG: v_accvgpr_write_b32 [[A_REG:a[0-9]+]], v{{[0-9]}}
; GFX900-NOT: buffer_
; GFX908-NOT: buffer_
; GFX908-DAG: v_mov_b32_e32 v{{[0-9]}}, [[V_REG:v[0-9]+]]
; GFX908-DAG: v_accvgpr_read_b32 [[V_REG]], [[A_REG]]

; GFX900: NumVgprs: 11
; GFX908: NumVgprs: 10
; GFX900: ScratchSize: 0
; GFX908: ScratchSize: 0
; GCN:    VGPRBlocks: 2
; GFX900: NumVGPRsForWavesPerEU: 11
; GFX908: NumVGPRsForWavesPerEU: 10
define amdgpu_kernel void @max_11_vgprs(ptr addrspace(1) %p) #0 {
  %tid = load volatile i32, ptr addrspace(1) poison
  %p1 = getelementptr inbounds i32, ptr addrspace(1) %p, i32 %tid
  %p2 = getelementptr inbounds i32, ptr addrspace(1) %p1, i32 4
  %p3 = getelementptr inbounds i32, ptr addrspace(1) %p2, i32 8
  %p4 = getelementptr inbounds i32, ptr addrspace(1) %p3, i32 12
  %p5 = getelementptr inbounds i32, ptr addrspace(1) %p4, i32 16
  %p6 = getelementptr inbounds i32, ptr addrspace(1) %p5, i32 20
  %p7 = getelementptr inbounds i32, ptr addrspace(1) %p6, i32 24
  %p8 = getelementptr inbounds i32, ptr addrspace(1) %p7, i32 28
  %p9 = getelementptr inbounds i32, ptr addrspace(1) %p8, i32 32
  %p10 = getelementptr inbounds i32, ptr addrspace(1) %p9, i32 36
  %v1 = load volatile i32, ptr addrspace(1) %p1
  %v2 = load volatile i32, ptr addrspace(1) %p2
  %v3 = load volatile i32, ptr addrspace(1) %p3
  %v4 = load volatile i32, ptr addrspace(1) %p4
  %v5 = load volatile i32, ptr addrspace(1) %p5
  %v6 = load volatile i32, ptr addrspace(1) %p6
  %v7 = load volatile i32, ptr addrspace(1) %p7
  %v8 = load volatile i32, ptr addrspace(1) %p8
  %v9 = load volatile i32, ptr addrspace(1) %p9
  %v10 = load volatile i32, ptr addrspace(1) %p10
  call void asm sideeffect "", "v,v,v,v,v,v,v,v,v,v"(i32 %v1, i32 %v2, i32 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10)
  store volatile i32 %v1, ptr addrspace(1) poison
  store volatile i32 %v2, ptr addrspace(1) poison
  store volatile i32 %v3, ptr addrspace(1) poison
  store volatile i32 %v4, ptr addrspace(1) poison
  store volatile i32 %v5, ptr addrspace(1) poison
  store volatile i32 %v6, ptr addrspace(1) poison
  store volatile i32 %v7, ptr addrspace(1) poison
  store volatile i32 %v8, ptr addrspace(1) poison
  store volatile i32 %v9, ptr addrspace(1) poison
  store volatile i32 %v10, ptr addrspace(1) poison
  ret void
}

attributes #0 = { nounwind }
