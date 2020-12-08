; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GCN,GFX908 %s
; RUN: not llc -march=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefixes=GCN,GFX900 %s

; GCN-LABEL: {{^}}max_10_vgprs:
; GFX900-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GFX900-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-NOT: SCRATCH_RSRC
; GFX908-DAG: v_accvgpr_write_b32 a0, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a1, v{{[0-9]}} ; Reload Reuse
; GFX900:     buffer_store_dword v{{[0-9]}},
; GFX900:     buffer_store_dword v{{[0-9]}},
; GFX900:     buffer_load_dword v{{[0-9]}},
; GFX900:     buffer_load_dword v{{[0-9]}},
; GFX908-NOT: buffer_
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a0 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a1 ; Reload Reuse

; GCN:    NumVgprs: 10
; GFX900: ScratchSize: 12
; GFX908: ScratchSize: 0
; GCN:    VGPRBlocks: 2
; GCN:    NumVGPRsForWavesPerEU: 10
define amdgpu_kernel void @max_10_vgprs(i32 addrspace(1)* %p) #0 {
  %tid = load volatile i32, i32 addrspace(1)* undef
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

; GCN-LABEL: {{^}}max_10_vgprs_used_9a:
; GFX908-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GFX908-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-DAG: v_accvgpr_write_b32 a9, v{{[0-9]}} ; Reload Reuse
; GFX908:     buffer_store_dword v{{[0-9]}},
; GFX908-NOT: buffer_
; GFX908:     v_accvgpr_read_b32 v{{[0-9]}}, a9 ; Reload Reuse
; GFX908:     buffer_load_dword v{{[0-9]}},
; GFX908-NOT: buffer_

; GFX900:     couldn't allocate input reg for constraint 'a'

; GFX908: NumVgprs: 10
; GFX908: ScratchSize: 8
; GFX908: VGPRBlocks: 2
; GFX908: NumVGPRsForWavesPerEU: 10
define amdgpu_kernel void @max_10_vgprs_used_9a(i32 addrspace(1)* %p) #0 {
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

; GCN-LABEL: {{^}}max_10_vgprs_used_1a_partial_spill:
; GCN-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-DAG: v_accvgpr_write_b32 a0, 1
; GFX908-DAG: v_accvgpr_write_b32 a1, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a2, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a3, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a4, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a5, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a6, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a7, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a8, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a9, v{{[0-9]}} ; Reload Reuse
; GFX900:     buffer_store_dword v{{[0-9]}},
; GCN-DAG:    buffer_store_dword v{{[0-9]}},
; GFX900:     buffer_load_dword v{{[0-9]}},
; GCN-DAG:    buffer_load_dword v{{[0-9]}},
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a1 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a2 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a3 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a4 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a5 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a6 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a7 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a8 ; Reload Reuse
; GFX908-DAG: v_accvgpr_read_b32 v{{[0-9]}}, a9 ; Reload Reuse

; GCN:    NumVgprs: 10
; GFX900: ScratchSize: 44
; GFX908: ScratchSize: 20
; GCN:    VGPRBlocks: 2
; GCN:    NumVGPRsForWavesPerEU: 10
define amdgpu_kernel void @max_10_vgprs_used_1a_partial_spill(i64 addrspace(1)* %p) #0 {
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

; GCN-LABEL: {{^}}max_10_vgprs_spill_v32:
; GCN-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-DAG: v_accvgpr_write_b32 a0, v{{[0-9]}} ; Reload Reuse
; GFX908-DAG: v_accvgpr_write_b32 a9, v{{[0-9]}} ; Reload Reuse
; GCN-NOT:    a10
; GCN:        buffer_store_dword v{{[0-9]}},

; GFX908: NumVgprs: 10
; GFX900: ScratchSize: 100
; GFX908: ScratchSize: 68
; GFX908: VGPRBlocks: 2
; GFX908: NumVGPRsForWavesPerEU: 10
define amdgpu_kernel void @max_10_vgprs_spill_v32(<32 x float> addrspace(1)* %p) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p, i32 %tid
  %v = load volatile <32 x float>, <32 x float> addrspace(1)* %gep
  store volatile <32 x float> %v, <32 x float> addrspace(1)* undef
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

; GCN:    NumVgprs: 256
; GFX900: ScratchSize: 148
; GFX908: ScratchSize: 0
; GCN:    VGPRBlocks: 63
; GCN:    NumVGPRsForWavesPerEU: 256
define amdgpu_kernel void @max_256_vgprs_spill_9x32(<32 x float> addrspace(1)* %p) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p1 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p, i32 %tid
  %p2 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p1, i32 %tid
  %p3 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p2, i32 %tid
  %p4 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p3, i32 %tid
  %p5 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p4, i32 %tid
  %p6 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p5, i32 %tid
  %p7 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p6, i32 %tid
  %p8 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p7, i32 %tid
  %p9 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p8, i32 %tid
  %v1 = load volatile <32 x float>, <32 x float> addrspace(1)* %p1
  %v2 = load volatile <32 x float>, <32 x float> addrspace(1)* %p2
  %v3 = load volatile <32 x float>, <32 x float> addrspace(1)* %p3
  %v4 = load volatile <32 x float>, <32 x float> addrspace(1)* %p4
  %v5 = load volatile <32 x float>, <32 x float> addrspace(1)* %p5
  %v6 = load volatile <32 x float>, <32 x float> addrspace(1)* %p6
  %v7 = load volatile <32 x float>, <32 x float> addrspace(1)* %p7
  %v8 = load volatile <32 x float>, <32 x float> addrspace(1)* %p8
  %v9 = load volatile <32 x float>, <32 x float> addrspace(1)* %p9
  store volatile <32 x float> %v1, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v2, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v3, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v4, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v5, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v6, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v7, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v8, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v9, <32 x float> addrspace(1)* undef
  ret void
}

; FIXME: adding an AReg_1024 register class for v32f32 and v32i32
;        produces unnecessary copies and we still have some amount
;        of conventional spilling.

; GCN-LABEL: {{^}}max_256_vgprs_spill_9x32_2bb:
; GFX900-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GFX900-DAG: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GFX908-FIXME-NOT: SCRATCH_RSRC
; GFX908-DAG: v_accvgpr_write_b32 a0, v
; GFX900:     buffer_store_dword v
; GFX900:     buffer_load_dword v
; GFX908-FIXME-NOT: buffer_
; GFX908-DAG: v_accvgpr_read_b32

; GCN:    NumVgprs: 256
; GFX900: ScratchSize: 2052
; GFX908-FIXME: ScratchSize: 0
; GCN:    VGPRBlocks: 63
; GCN:    NumVGPRsForWavesPerEU: 256
define amdgpu_kernel void @max_256_vgprs_spill_9x32_2bb(<32 x float> addrspace(1)* %p) #1 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %p1 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p, i32 %tid
  %p2 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p1, i32 %tid
  %p3 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p2, i32 %tid
  %p4 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p3, i32 %tid
  %p5 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p4, i32 %tid
  %p6 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p5, i32 %tid
  %p7 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p6, i32 %tid
  %p8 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p7, i32 %tid
  %p9 = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %p8, i32 %tid
  %v1 = load volatile <32 x float>, <32 x float> addrspace(1)* %p1
  %v2 = load volatile <32 x float>, <32 x float> addrspace(1)* %p2
  %v3 = load volatile <32 x float>, <32 x float> addrspace(1)* %p3
  %v4 = load volatile <32 x float>, <32 x float> addrspace(1)* %p4
  %v5 = load volatile <32 x float>, <32 x float> addrspace(1)* %p5
  %v6 = load volatile <32 x float>, <32 x float> addrspace(1)* %p6
  %v7 = load volatile <32 x float>, <32 x float> addrspace(1)* %p7
  %v8 = load volatile <32 x float>, <32 x float> addrspace(1)* %p8
  %v9 = load volatile <32 x float>, <32 x float> addrspace(1)* %p9
  br label %st

st:
  store volatile <32 x float> %v1, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v2, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v3, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v4, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v5, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v6, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v7, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v8, <32 x float> addrspace(1)* undef
  store volatile <32 x float> %v9, <32 x float> addrspace(1)* undef
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { nounwind "amdgpu-num-vgpr"="10" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,256" }
