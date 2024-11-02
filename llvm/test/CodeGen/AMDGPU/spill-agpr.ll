; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX908 %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN,GFX90A %s

; GCN-LABEL: {{^}}max_12regs_13a_used:
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN:     v_accvgpr_read_b32 v[[VSPILL:[0-9]+]], a{{[0-9]+}}
; GCN-NOT: buffer_store_dword
; GCN-NOT: buffer_load_dword
; GCN:     v_accvgpr_write_b32 a{{[0-9]+}}, v[[VSPILL]]
; GCN:     ScratchSize: 0
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
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN:     v_accvgpr_read_b32 v[[VSPILL:[0-9]+]], a{{[0-9]+}}
; GCN-NOT: buffer_store_dword
; GCN-NOT: buffer_load_dword
; GCN:     v_accvgpr_write_b32 a{{[0-9]+}}, v[[VSPILL]]
; GCN:     ScratchSize: 0
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
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN-NOT: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1
; GCN-NOT: buffer_store_dword
; GCN:     v_accvgpr_read_b32
; GCN:     v_mfma_f32_32x32x1f32
; GCN-NOT: buffer_load_dword
; GCN:     v_accvgpr_write_b32
; GCN:     ScratchSize: 0
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

; Should spill agprs to memory for both gfx908 and gfx90a.
; GCN-LABEL: {{^}}max_6regs_used_8a:
; GCN: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD0
; GCN: s_mov_b32 s{{[0-9]+}}, SCRATCH_RSRC_DWORD1

; GFX908-DAG:  v_accvgpr_read_b32 v5, a0 ; Reload Reuse
; GFX908-DAG:  buffer_store_dword v5, off, s[{{[0-9:]+}}], 0 offset:4 ; 4-byte Folded Spill
; GFX908-DAG:  v_accvgpr_read_b32 v5, a1 ; Reload Reuse
; GFX908-DAG:  buffer_store_dword v5, off, s[{{[0-9:]+}}], 0 offset:8 ; 4-byte Folded Spill
; GFX908-DAG:  v_accvgpr_read_b32 v5, a2 ; Reload Reuse
; GFX908-DAG:  buffer_store_dword v5, off, s[{{[0-9:]+}}], 0 offset:12 ; 4-byte Folded Spill
; GFX908-DAG:  v_accvgpr_read_b32 v5, a3 ; Reload Reuse
; GFX908-DAG:  buffer_store_dword v5, off, s[{{[0-9:]+}}], 0 offset:16 ; 4-byte Folded Spill

; GFX90A-DAG:  buffer_store_dword a0, off, s[{{[0-9:]+}}], 0 offset:4 ; 4-byte Folded Spill
; GFX90A-DAG:  buffer_store_dword a1, off, s[{{[0-9:]+}}], 0 offset:8 ; 4-byte Folded Spill
; GFX90A-DAG:  buffer_store_dword a2, off, s[{{[0-9:]+}}], 0 offset:12 ; 4-byte Folded Spill
; GFX90A-DAG:  buffer_store_dword a3, off, s[{{[0-9:]+}}], 0 offset:16 ; 4-byte Folded Spill

; GCN:  v_mfma_f32_4x4x1f32 a[0:3], v{{[0-9]+}}, v{{[0-9]+}}, a[0:3]

; GFX908-DAG:  buffer_load_dword v0, off, s[{{[0-9:]+}}], 0 offset:4 ; 4-byte Folded Reload
; GFX908-DAG:  buffer_load_dword v1, off, s[{{[0-9:]+}}], 0 offset:8 ; 4-byte Folded Reload
; GFX908-DAG:  buffer_load_dword v2, off, s[{{[0-9:]+}}], 0 offset:12 ; 4-byte Folded Reload
; GFX908-DAG:  buffer_load_dword v3, off, s[{{[0-9:]+}}], 0 offset:16 ; 4-byte Folded Reload
; GFX908: global_store_dwordx4 v[{{[0-9:]+}}], v[0:3], off

; GFX90A-DAG:  buffer_load_dword v2, off, s[4:7], 0 offset:4 ; 4-byte Folded Reload
; GFX90A-DAG:  buffer_load_dword v3, off, s[4:7], 0 offset:8 ; 4-byte Folded Reload
; GFX90A-DAG:  buffer_load_dword v4, off, s[4:7], 0 offset:12 ; 4-byte Folded Reload
; GFX90A-DAG:  buffer_load_dword v5, off, s[4:7], 0 offset:16 ; 4-byte Folded Reload
; GFX90A:  global_store_dwordx4 v[0:1], v[2:5], off

; GCN: ScratchSize: 20
define amdgpu_kernel void @max_6regs_used_8a(<4 x float> addrspace(1)* %arg) #4 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %v0 = call float asm sideeffect "; def $0", "=v"()
  %a4 = call <4 x float> asm sideeffect "; def $0", "=a"()
  %gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %arg, i32 %tid
  %mai.in = load <4 x float>, <4 x float> addrspace(1)* %gep
  %mai.out = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 1.0, <4 x float> %mai.in, i32 0, i32 0, i32 0)
  store <4 x float> %mai.out, <4 x float> addrspace(1)* %gep
  store volatile <4 x float> %a4, <4 x float> addrspace(1)* undef
  call void asm sideeffect "; use $0", "v"(float %v0);
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)

attributes #1 = { nounwind "amdgpu-num-vgpr"="10" }
attributes #2 = { nounwind "amdgpu-num-vgpr"="12" }
attributes #3 = { nounwind "amdgpu-num-vgpr"="32" }
attributes #4 = { nounwind "amdgpu-num-vgpr"="6" }
