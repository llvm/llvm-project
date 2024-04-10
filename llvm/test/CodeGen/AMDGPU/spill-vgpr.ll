; RUN: llc -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX908 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX900 %s

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
define amdgpu_kernel void @max_11_vgprs(ptr addrspace(1) %p) #2 {
  %tid = load volatile i32, ptr addrspace(1) undef
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
  store volatile i32 %v1, ptr addrspace(1) undef
  store volatile i32 %v2, ptr addrspace(1) undef
  store volatile i32 %v3, ptr addrspace(1) undef
  store volatile i32 %v4, ptr addrspace(1) undef
  store volatile i32 %v5, ptr addrspace(1) undef
  store volatile i32 %v6, ptr addrspace(1) undef
  store volatile i32 %v7, ptr addrspace(1) undef
  store volatile i32 %v8, ptr addrspace(1) undef
  store volatile i32 %v9, ptr addrspace(1) undef
  store volatile i32 %v10, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}max_10_vgprs_spill_v32:
; GCN-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN:        buffer_store_dword v{{[0-9]}},
; GFX908-DAG: v_accvgpr_write_b32 a0, v{{[0-9]}}
; GFX908-DAG: v_accvgpr_write_b32 a9, v{{[0-9]}}
; GCN-NOT:    a10

; GFX908: NumVgprs: 10
; GFX900: ScratchSize: 100
; GFX908: ScratchSize: 68
; GFX908: VGPRBlocks: 2
; GFX908: NumVGPRsForWavesPerEU: 10
define amdgpu_kernel void @max_10_vgprs_spill_v32(ptr addrspace(1) %p) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, ptr addrspace(1) %p, i32 %tid
  %v = load volatile <32 x float>, ptr addrspace(1) %gep
  store volatile <32 x float> %v, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}max_256_vgprs_spill_9x32:
; GFX900-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GFX900-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-NOT: SCRATCH_RSRC
; GFX908-DAG: v_accvgpr_write_b32 a0, v
; GFX900:     buffer_store_dword v
; GFX900:     buffer_load_dword v
; GFX908-NOT: buffer_
; GFX908-DAG: v_accvgpr_read_b32

; GFX900: NumVgprs: 256
; GFX900: ScratchSize: 132
; GFX908: NumVgprs: 252
; GFX908: ScratchSize: 0
; GCN900:    VGPRBlocks: 63
; GCN908:    VGPRBlocks: 62
; GFX900:    NumVGPRsForWavesPerEU: 256
; GFX908:    NumVGPRsForWavesPerEU: 252
define amdgpu_kernel void @max_256_vgprs_spill_9x32(ptr addrspace(1) %p) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p1 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p, i32 %tid
  %p2 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p1, i32 %tid
  %p3 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p2, i32 %tid
  %p4 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p3, i32 %tid
  %p5 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p4, i32 %tid
  %p6 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p5, i32 %tid
  %p7 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p6, i32 %tid
  %p8 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p7, i32 %tid
  %p9 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p8, i32 %tid
  %v1 = load volatile <32 x float>, ptr addrspace(1) %p1
  %v2 = load volatile <32 x float>, ptr addrspace(1) %p2
  %v3 = load volatile <32 x float>, ptr addrspace(1) %p3
  %v4 = load volatile <32 x float>, ptr addrspace(1) %p4
  %v5 = load volatile <32 x float>, ptr addrspace(1) %p5
  %v6 = load volatile <32 x float>, ptr addrspace(1) %p6
  %v7 = load volatile <32 x float>, ptr addrspace(1) %p7
  %v8 = load volatile <32 x float>, ptr addrspace(1) %p8
  %v9 = load volatile <32 x float>, ptr addrspace(1) %p9
  store volatile <32 x float> %v1, ptr addrspace(1) undef
  store volatile <32 x float> %v2, ptr addrspace(1) undef
  store volatile <32 x float> %v3, ptr addrspace(1) undef
  store volatile <32 x float> %v4, ptr addrspace(1) undef
  store volatile <32 x float> %v5, ptr addrspace(1) undef
  store volatile <32 x float> %v6, ptr addrspace(1) undef
  store volatile <32 x float> %v7, ptr addrspace(1) undef
  store volatile <32 x float> %v8, ptr addrspace(1) undef
  store volatile <32 x float> %v9, ptr addrspace(1) undef
  ret void
}

; GCN-LABEL: {{^}}max_256_vgprs_spill_9x32_2bb:
; GFX900-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GFX900-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-NOT: SCRATCH_RSRC
; GFX908: v_accvgpr_write_b32
; GFX908:  global_load_
; GFX900:     buffer_store_dword v
; GFX900:     buffer_load_dword v
; GFX908-NOT: buffer_
; GFX908-DAG: v_accvgpr_read_b32

; GFX900: NumVgprs: 256
; GFX908: NumVgprs: 252
; GFX900: ScratchSize: 132
; GFX908: ScratchSize: 0
; GFX900: VGPRBlocks: 63
; GFX908: VGPRBlocks: 62
; GFX900: NumVGPRsForWavesPerEU: 256
; GFX908: NumVGPRsForWavesPerEU: 252
define amdgpu_kernel void @max_256_vgprs_spill_9x32_2bb(ptr addrspace(1) %p) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p1 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p, i32 %tid
  %p2 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p1, i32 %tid
  %p3 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p2, i32 %tid
  %p4 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p3, i32 %tid
  %p5 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p4, i32 %tid
  %p6 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p5, i32 %tid
  %p7 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p6, i32 %tid
  %p8 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p7, i32 %tid
  %p9 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p8, i32 %tid
  %v1 = load volatile <32 x float>, ptr addrspace(1) %p1
  %v2 = load volatile <32 x float>, ptr addrspace(1) %p2
  %v3 = load volatile <32 x float>, ptr addrspace(1) %p3
  %v4 = load volatile <32 x float>, ptr addrspace(1) %p4
  %v5 = load volatile <32 x float>, ptr addrspace(1) %p5
  %v6 = load volatile <32 x float>, ptr addrspace(1) %p6
  %v7 = load volatile <32 x float>, ptr addrspace(1) %p7
  %v8 = load volatile <32 x float>, ptr addrspace(1) %p8
  %v9 = load volatile <32 x float>, ptr addrspace(1) %p9
  br label %st

st:
  store volatile <32 x float> %v1, ptr addrspace(1) undef
  store volatile <32 x float> %v2, ptr addrspace(1) undef
  store volatile <32 x float> %v3, ptr addrspace(1) undef
  store volatile <32 x float> %v4, ptr addrspace(1) undef
  store volatile <32 x float> %v5, ptr addrspace(1) undef
  store volatile <32 x float> %v6, ptr addrspace(1) undef
  store volatile <32 x float> %v7, ptr addrspace(1) undef
  store volatile <32 x float> %v8, ptr addrspace(1) undef
  store volatile <32 x float> %v9, ptr addrspace(1) undef
  ret void
}

; Make sure there's no crash when we have loads from fixed stack
; objects and are processing VGPR spills

; GCN-LABEL: {{^}}stack_args_vgpr_spill:
; GFX908: v_accvgpr_write_b32
; GFX908: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32
; GFX908: buffer_load_dword v{{[0-9]+}}, off, s[0:3], s32 offset:4
define void @stack_args_vgpr_spill(<32 x float> %arg0, <32 x float> %arg1, ptr addrspace(1) %p) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p1 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p, i32 %tid
  %p2 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p1, i32 %tid
  %p3 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p2, i32 %tid
  %p4 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p3, i32 %tid
  %p5 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p4, i32 %tid
  %p6 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p5, i32 %tid
  %p7 = getelementptr inbounds <32 x float>, ptr addrspace(1) %p6, i32 %tid
  %v1 = load volatile <32 x float>, ptr addrspace(1) %p1
  %v2 = load volatile <32 x float>, ptr addrspace(1) %p2
  %v3 = load volatile <32 x float>, ptr addrspace(1) %p3
  %v4 = load volatile <32 x float>, ptr addrspace(1) %p4
  %v5 = load volatile <32 x float>, ptr addrspace(1) %p5
  %v6 = load volatile <32 x float>, ptr addrspace(1) %p6
  %v7 = load volatile <32 x float>, ptr addrspace(1) %p7
  br label %st

st:
  store volatile <32 x float> %arg0, ptr addrspace(1) undef
  store volatile <32 x float> %arg1, ptr addrspace(1) undef
  store volatile <32 x float> %v1, ptr addrspace(1) undef
  store volatile <32 x float> %v2, ptr addrspace(1) undef
  store volatile <32 x float> %v3, ptr addrspace(1) undef
  store volatile <32 x float> %v4, ptr addrspace(1) undef
  store volatile <32 x float> %v5, ptr addrspace(1) undef
  store volatile <32 x float> %v6, ptr addrspace(1) undef
  store volatile <32 x float> %v7, ptr addrspace(1) undef
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { nounwind "amdgpu-num-vgpr"="10" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,256" }
attributes #2 = { nounwind "amdgpu-num-vgpr"="11" }
