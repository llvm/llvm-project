; RUN: llc -mtriple=amdgpu9.08 -amdgpu-stress-vgpr=10 -amdgpu-stress-agpr=10 < %s | FileCheck -check-prefixes=GCN,GFX908 %s
; RUN: llc -mtriple=amdgpu9.00 -amdgpu-stress-vgpr=10 < %s | FileCheck -check-prefixes=GCN,GFX900 %s

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
  store volatile <32 x float> %v, ptr addrspace(1) poison
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()

attributes #0 = { nounwind }
