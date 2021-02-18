; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX908,A2V %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -amdgpu-spill-vgpr-to-agpr=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX908,GFX908-A2M,A2M %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -amdgpu-spill-vgpr-to-agpr=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX90A,GFX90A-A2M,A2M %s

; GCN-LABEL: {{^}}max_24regs_32a_used:
; A2M-DAG:        s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; A2M-DAG:        s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN-DAG:        v_mfma_f32_16x16x1f32
; GCN-DAG:        v_mfma_f32_16x16x1f32
; A2V-NOT:        SCRATCH_RSRC
; GFX908-A2M-DAG: v_accvgpr_read_b32 v[[VSPILL:[0-9]+]], a0 ; Reload Reuse
; A2V:            v_accvgpr_read_b32 v[[VSPILL:[0-9]+]], a0 ; Reload Reuse
; GFX908-A2M:     buffer_store_dword v[[VSPILL]], off, s[{{[0-9:]+}}], 0 offset:[[FI:[0-9]+]] ; 4-byte Folded Spill
; GFX908-A2M:     buffer_load_dword v[[VSPILL:[0-9]+]], off, s[{{[0-9:]+}}], 0 offset:[[FI]] ; 4-byte Folded Reload
; GFX90A-NOT:     v_accvgpr_read_b32
; GFX90A-A2M:     buffer_store_dword a{{[0-9]+}}, off, s[{{[0-9:]+}}], 0 offset:[[FI:[0-9]+]] ; 4-byte Folded Spill
; GFX90A-A2M:     buffer_load_dword a{{[0-9]+}}, off, s[{{[0-9:]+}}], 0 offset:[[FI]] ; 4-byte Folded Reload
; GFX908:         v_accvgpr_write_b32 a{{[0-9]+}}, v[[VSPILL]] ; Reload Reuse
; GFX90A-NOT:     v_accvgpr_write_b32
; A2V:            ScratchSize: 0
define amdgpu_kernel void @max_24regs_32a_used(<16 x float> addrspace(1)* %arg, float addrspace(1)* %out) #0 {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 1.0, <16 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 1.0, <16 x float> %mai.1, i32 0, i32 0, i32 0)
  %elt1 = extractelement <16 x float> %mai.2, i32 0
  %elt2 = extractelement <16 x float> %mai.1, i32 15
  %elt3 = extractelement <16 x float> %mai.1, i32 14
  %elt4 = extractelement <16 x float> %mai.2, i32 1
  store float %elt1, float addrspace(1)* %out
  %gep1 = getelementptr float, float addrspace(1)* %out, i64 1
  store float %elt2, float addrspace(1)* %gep1
  %gep2 = getelementptr float, float addrspace(1)* %out, i64 2
  store float %elt3, float addrspace(1)* %gep2
  %gep3 = getelementptr float, float addrspace(1)* %out, i64 3
  store float %elt4, float addrspace(1)* %gep3

  ret void
}

; GCN-LABEL: {{^}}max_12regs_13a_used:
; A2M-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; A2M-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; A2V-NOT:    SCRATCH_RSRC
; GFX908-DAG: v_accvgpr_read_b32 v[[VSPILL:[0-9]+]], a{{[0-9]+}} ; Reload Reuse
; GFX908-A2M: buffer_store_dword v[[VSPILL]], off, s[{{[0-9:]+}}], 0 offset:[[FI:[0-9]+]] ; 4-byte Folded Spill
; GFX908-A2M: buffer_load_dword v[[VSPILL:[0-9]+]], off, s[{{[0-9:]+}}], 0 offset:[[FI]] ; 4-byte Folded Reload
; GFX90A-A2M: buffer_store_dword a{{[0-9]+}}, off, s[{{[0-9:]+}}], 0 offset:[[FI:[0-9]+]] ; 4-byte Folded Spill
; GFX90A-A2M: buffer_load_dword a{{[0-9]+}}, off, s[{{[0-9:]+}}], 0 offset:[[FI]] ; 4-byte Folded Reload
; A2V:        v_accvgpr_write_b32 a{{[0-9]+}}, v[[VSPILL]] ; Reload Reuse
; A2V:        ScratchSize: 0
define amdgpu_kernel void @max_12regs_13a_used(i32 %cond, <4 x float> addrspace(1)* %arg, <4 x float> addrspace(1)* %out) #2 {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 1.0, <4 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 1.0, <4 x float> %mai.1, i32 0, i32 0, i32 0)
  %cmp = icmp eq i32 %cond, 0
  br i1 %cmp, label %use, label %st

use:
  call void asm sideeffect "", "a,a,a,a,a"(i32 1, i32 2, i32 3, i32 4, i32 5)
  store volatile <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, <4 x float> addrspace(1)* %out
  br label %st

st:
  %gep1 = getelementptr <4 x float>, <4 x float> addrspace(1)* %out, i64 16
  %gep2 = getelementptr <4 x float>, <4 x float> addrspace(1)* %out, i64 32
  call void asm sideeffect "", "a,a"(<4 x float> %mai.1, <4 x float> %mai.2)
  ret void
}

; GCN-LABEL: {{^}}max_10_vgprs_used_9a:
; GFX908-A2M-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GFX908-A2M-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; A2V-NOT:    SCRATCH_RSRC

; A2V: v_accvgpr_read_b32 v[[VSPILL:[0-9]+]], a{{[0-9]+}} ; Reload Reuse
; A2V: v_accvgpr_write_b32 a{{[0-9]+}}, v[[VSPILL]] ; Reload Reuse
; A2V: ScratchSize: 0

; GFX908-A2M: buffer_store_dword v[[VSPILLSTORE:[0-9]+]], off, s[{{[0-9:]+}}], 0 offset:[[FI:[0-9]+]] ; 4-byte Folded Spill
; GFX908-A2M: buffer_load_dword v[[VSPILL_RELOAD:[0-9]+]], off, s[{{[0-9:]+}}], 0 offset:[[FI]] ; 4-byte Folded Reload
; GFX908-A2M: v_accvgpr_write_b32 a{{[0-9]+}}, v[[VSPILL_RELOAD]] ; Reload Reuse
define amdgpu_kernel void @max_10_vgprs_used_9a() #1 {
  %a1 = call <4 x i32> asm sideeffect "", "=a"()
  %a2 = call <4 x i32> asm sideeffect "", "=a"()
  %a3 = call i32 asm sideeffect "", "=a"()
  %a4 = call <2 x i32> asm sideeffect "", "=a"()
  call void asm sideeffect "", "a,a,a"(<4 x i32> %a1, <4 x i32> %a2, i32 %a3)
  call void asm sideeffect "", "a"(<2 x i32> %a4)
  ret void
}

; GCN-LABEL: {{^}}max_32regs_mfma32:
; A2M-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; A2M-DAG:    s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; A2V-NOT:    SCRATCH_RSRC
; GFX908-DAG: v_accvgpr_read_b32 v[[VSPILL:[0-9]+]], a0 ; Reload Reuse
; GFX908-A2M: buffer_store_dword v[[VSPILL]], off, s[{{[0-9:]+}}], 0 offset:[[FI:[0-9]+]] ; 4-byte Folded Spill
; GFX90A-NOT: v_accvgpr_read_b32
; GFX90A:     v_mfma_f32_32x32x1f32
; GFX908-A2M: buffer_load_dword v[[VSPILL:[0-9]+]], off, s[{{[0-9:]+}}], 0 offset:[[FI]] ; 4-byte Folded Reload
; GFX908:     v_accvgpr_write_b32 a{{[0-9]+}}, v[[VSPILL]] ; Reload Reuse
; GFX90A-NOT: v_accvgpr_write_b32
; A2V:        ScratchSize: 0
define amdgpu_kernel void @max_32regs_mfma32(float addrspace(1)* %arg) #3 {
bb:
  %v = call i32 asm sideeffect "", "=a"()
  br label %use

use:
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 1.0, <32 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0, float 17.0, float 18.0, float 19.0, float 20.0, float 21.0, float 22.0, float 23.0, float 24.0, float 25.0, float 26.0, float 27.0, float 28.0, float 29.0, float 30.0, float 31.0, float 2.0>, i32 0, i32 0, i32 0)
  call void asm sideeffect "", "a"(i32 %v)
  %elt1 = extractelement <32 x float> %mai.1, i32 0
  store float %elt1, float addrspace(1)* %arg
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)

attributes #0 = { nounwind "amdgpu-num-vgpr"="24" }
attributes #1 = { nounwind "amdgpu-num-vgpr"="10" }
attributes #2 = { nounwind "amdgpu-num-vgpr"="12" }
attributes #3 = { nounwind "amdgpu-num-vgpr"="32" }
