; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX940 %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -global-isel -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GISEL %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -stress-regalloc=10 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX940 %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -stress-regalloc=10 -global-isel -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GISEL %s

declare <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64, i64, <4 x i32>, i32, i32, i32)
declare <16 x i32> @llvm.amdgcn.mfma.i32.32x32x16.i8(i64, i64, <16 x i32>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x8.xf32(<2 x float>, <2 x float>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x4.xf32(<2 x float>, <2 x float>, <16 x float>, i32, i32, i32)

; GCN-LABEL: {{^}}test_mfma_i32_16x16x32i8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_i32_16x16x32_i8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_i32_16x16x32_i8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_i32_16x16x32i8(<4 x i32> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <4 x i32>, <4 x i32> addrspace(1)* %arg
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 4294967298, i64 12884901892, <4 x i32> %in.1, i32 1, i32 2, i32 3)
  store <4 x i32> %mai.1, <4 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_32x32x16i8:
; GFX940-DAG:   v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:   v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:   v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:   v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-16: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:       v_mfma_i32_32x32x16_i8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:        v_mfma_i32_32x32x16_i8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:      v_accvgpr_read_b32
; GCN-COUNT-4:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_i32_32x32x16i8(<16 x i32> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <16 x i32>, <16 x i32> addrspace(1)* %arg
  %mai.1 = tail call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x16.i8(i64 4294967298, i64 12884901892, <16 x i32> %in.1, i32 1, i32 2, i32 3)
  store <16 x i32> %mai.1, <16 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x8xf32:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1.0
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2.0
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 0x40400000
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4.0
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_16x16x8_xf32 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[ONE]]:[[TWO]]], v{{\[}}[[THREE]]:[[FOUR]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_16x16x8_xf32 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x8xf32(<4 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8.xf32(<2 x float> <float 1.0, float 2.0>, <2 x float> <float 3.0, float 4.0>, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x4xf32:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1.0
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2.0
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 0x40400000
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4.0
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_32x32x4_xf32 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[ONE]]:[[TWO]]], v{{\[}}[[THREE]]:[[FOUR]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_32x32x4_xf32 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x4xf32(<16 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x4.xf32(<2 x float> <float 1.0, float 2.0>, <2 x float> <float 3.0, float 4.0>, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
