; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX940,VGPRCD %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -global-isel -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GISEL,VGPRCD %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -stress-regalloc=10 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX940,AGPRCD %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -stress-regalloc=10 -global-isel -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GISEL,AGPRCD %s

declare <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64, i64, <4 x i32>, i32, i32, i32)
declare <16 x i32> @llvm.amdgcn.mfma.i32.32x32x16.i8(i64, i64, <16 x i32>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x8.xf32(<2 x float>, <2 x float>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x4.xf32(<2 x float>, <2 x float>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8(i64, i64, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64, i64, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8(i64, i64, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64, i64, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64, i64, <16 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8(i64, i64, <16 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8(i64, i64, <16 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8(i64, i64, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x32.f16(<4 x half>, <8 x half>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x16.f16(<4 x half>, <8 x half>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x32.bf16(<4 x i16>, <8 x i16>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x16.bf16(<4 x i16>, <8 x i16>, <16 x float>, i32, i32, i32)
declare <4 x i32> @llvm.amdgcn.smfmac.i32.16x16x64.i8(<2 x i32>, <4 x i32>, <4 x i32>, i32, i32, i32)
declare <16 x i32> @llvm.amdgcn.smfmac.i32.32x32x32.i8(<2 x i32>, <4 x i32>, <16 x i32>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf8.bf8(<2 x i32>, <4 x i32>, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf8.fp8(<2 x i32>, <4 x i32>, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.fp8.bf8(<2 x i32>, <4 x i32>, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.fp8.fp8(<2 x i32>, <4 x i32>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.bf8.bf8(<2 x i32>, <4 x i32>, <16 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.bf8.fp8(<2 x i32>, <4 x i32>, <16 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.fp8.bf8(<2 x i32>, <4 x i32>, <16 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.fp8.fp8(<2 x i32>, <4 x i32>, <16 x float>, i32, i32, i32)

; GCN-LABEL: {{^}}test_mfma_i32_16x16x32i8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_i32_16x16x32_i8 a[{{[0-9]+:[0-9]+}}], v[[[TWO]]:[[ONE]]], v[[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_i32_16x16x32_i8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_i32_16x16x32i8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x32.i8(i64 4294967298, i64 12884901892, <4 x i32> %in.1, i32 1, i32 2, i32 3)
  store <4 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_32x32x16i8:
; GFX940-DAG:   v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:   v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:   v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:   v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-16: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:       v_mfma_i32_32x32x16_i8 a[{{[0-9]+:[0-9]+}}], v[[[TWO]]:[[ONE]]], v[[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:        v_mfma_i32_32x32x16_i8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:      v_accvgpr_read_b32
; GCN-COUNT-4:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_i32_32x32x16i8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x16.i8(i64 4294967298, i64 12884901892, <16 x i32> %in.1, i32 1, i32 2, i32 3)
  store <16 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x8xf32:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1.0
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2.0
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 0x40400000
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4.0
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_16x16x8_xf32 a[{{[0-9]+:[0-9]+}}], v[[[ONE]]:[[TWO]]], v[[[THREE]]:[[FOUR]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_16x16x8_xf32 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x8xf32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8.xf32(<2 x float> <float 1.0, float 2.0>, <2 x float> <float 3.0, float 4.0>, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x4xf32:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1.0
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2.0
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 0x40400000
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4.0
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_32x32x4_xf32 a[{{[0-9]+:[0-9]+}}], v[[[ONE]]:[[TWO]]], v[[[THREE]]:[[FOUR]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_32x32x4_xf32 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x4xf32(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x4.xf32(<2 x float> <float 1.0, float 2.0>, <2 x float> <float 3.0, float 4.0>, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x32_bf8_bf8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_16x16x32_bf8_bf8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_16x16x32_bf8_bf8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x32_bf8_bf8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8(i64 4294967298, i64 12884901892, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x32_bf8_fp8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_16x16x32_bf8_fp8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_16x16x32_bf8_fp8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x32_bf8_fp8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8(i64 4294967298, i64 12884901892, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x32_fp8_bf8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_16x16x32_fp8_bf8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_16x16x32_fp8_bf8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x32_fp8_bf8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8(i64 4294967298, i64 12884901892, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x32_fp8_fp8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_16x16x32_fp8_fp8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_16x16x32_fp8_fp8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x32_fp8_fp8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8(i64 4294967298, i64 12884901892, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x16_bf8_bf8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_32x32x16_bf8_bf8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_32x32x16_bf8_bf8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x16_bf8_bf8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.bf8(i64 4294967298, i64 12884901892, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x16_bf8_fp8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_32x32x16_bf8_fp8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_32x32x16_bf8_fp8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x16_bf8_fp8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8(i64 4294967298, i64 12884901892, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x16_fp8_bf8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_32x32x16_fp8_bf8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_32x32x16_fp8_bf8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x16_fp8_bf8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8(i64 4294967298, i64 12884901892, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x16_fp8_fp8:
; GFX940-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX940-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX940-DAG:  v_mov_b32_e32 v[[THREE:[0-9]+]], 3
; GFX940-DAG:  v_mov_b32_e32 v[[FOUR:[0-9]+]], 4
; GCN-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX940:      v_mfma_f32_32x32x16_fp8_fp8 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[TWO]]:[[ONE]]], v{{\[}}[[FOUR]]:[[THREE]]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GISEL:       v_mfma_f32_32x32x16_fp8_fp8 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN:         global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x16_fp8_fp8(ptr addrspace(1) %arg) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x16.fp8.fp8(i64 4294967298, i64 12884901892, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_f32_16x16x32_f16:
; GCN:        s_load_dwordx4 s[[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]][[[RLO:[0-9]+]]:{{[0-9]+}}], s[[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_16x16x32_f16 [[CD]][[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN:        global_store_dwordx4 v{{[0-9]+}}, [[CD]][[[RLO]]:[[RHI]]]
define amdgpu_kernel void @test_smfmac_f32_16x16x32_f16(ptr addrspace(1) %arg, <4 x half> %a, <8 x half> %b, i32 %idx) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x32.f16(<4 x half> %a, <8 x half> %b, <4 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_f32_32x32x16_f16:
; GCN:        s_load_dwordx16 s[[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]][[[RLO:[0-9]+]]:{{[0-9]+}}], s[[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_32x32x16_f16 [[CD]][[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][[[RLO]]:{{[0-9]+}}], s[{{[0-9:]+}}]{{$}}
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:16
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:32
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9]+}}:[[RHI]]], s[{{[0-9:]+}}] offset:48
define amdgpu_kernel void @test_smfmac_f32_32x32x16_f16(ptr addrspace(1) %arg, <4 x half> %a, <8 x half> %b, i32 %idx) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x16.f16(<4 x half> %a, <8 x half> %b, <16 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_f32_16x16x32_bf16:
; GCN:        s_load_dwordx4 s[[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]][[[RLO:[0-9]+]]:{{[0-9]+}}], s[[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_16x16x32_bf16 [[CD]][[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN:        global_store_dwordx4 v{{[0-9]+}}, [[CD]][[[RLO]]:[[RHI]]]
define amdgpu_kernel void @test_smfmac_f32_16x16x32_bf16(ptr addrspace(1) %arg, <4 x i16> %a, <8 x i16> %b, i32 %idx) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x32.bf16(<4 x i16> %a, <8 x i16> %b, <4 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_f32_32x32x16_bf16:
; GCN:        s_load_dwordx16 s[[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]][[[RLO:[0-9]+]]:{{[0-9]+}}], s[[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_32x32x16_bf16 [[CD]][[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][[[RLO]]:{{[0-9]+}}], s[{{[0-9:]+}}]{{$}}
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:16
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:32
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9]+}}:[[RHI]]], s[{{[0-9:]+}}] offset:48
define amdgpu_kernel void @test_smfmac_f32_32x32x16_bf16(ptr addrspace(1) %arg, <4 x i16> %a, <8 x i16> %b, i32 %idx) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x16.bf16(<4 x i16> %a, <8 x i16> %b, <16 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_16x16x64_i8:
; GCN:        s_load_dwordx4 s[[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]][[[RLO:[0-9]+]]:{{[0-9]+}}], s[[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_i32_16x16x64_i8 [[CD]][[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN:        global_store_dwordx4 v{{[0-9]+}}, [[CD]][[[RLO]]:[[RHI]]]
define amdgpu_kernel void @test_smfmac_i32_16x16x64_i8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <4 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.smfmac.i32.16x16x64.i8(<2 x i32> %a, <4 x i32> %b, <4 x i32> %in.1, i32 %idx, i32 1, i32 2)
  store <4 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_32x32x32_i8:
; GCN:        s_load_dwordx16 s[[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]][[[RLO:[0-9]+]]:{{[0-9]+}}], s[[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_i32_32x32x32_i8 [[CD]][[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][[[RLO]]:{{[0-9]+}}], s[{{[0-9:]+}}]{{$}}
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:16
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:32
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9]+}}:[[RHI]]], s[{{[0-9:]+}}] offset:48
define amdgpu_kernel void @test_smfmac_i32_32x32x32_i8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <16 x i32>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x i32> @llvm.amdgcn.smfmac.i32.32x32x32.i8(<2 x i32> %a, <4 x i32> %b, <16 x i32> %in.1, i32 %idx, i32 1, i32 2)
  store <16 x i32> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_16x16x64_bf8_bf8:
; GCN:        s_load_dwordx4 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]]{{\[}}[[RLO:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_16x16x64_bf8_bf8 [[CD]]{{\[}}[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN:        global_store_dwordx4 v{{[0-9]+}}, [[CD]]{{\[}}[[RLO]]:[[RHI]]]
define amdgpu_kernel void @test_smfmac_i32_16x16x64_bf8_bf8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf8.bf8(<2 x i32> %a, <4 x i32> %b, <4 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_16x16x64_bf8_fp8:
; GCN:        s_load_dwordx4 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]]{{\[}}[[RLO:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_16x16x64_bf8_fp8 [[CD]]{{\[}}[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN:        global_store_dwordx4 v{{[0-9]+}}, [[CD]]{{\[}}[[RLO]]:[[RHI]]]
define amdgpu_kernel void @test_smfmac_i32_16x16x64_bf8_fp8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.bf8.fp8(<2 x i32> %a, <4 x i32> %b, <4 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_16x16x64_fp8_bf8:
; GCN:        s_load_dwordx4 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]]{{\[}}[[RLO:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_16x16x64_fp8_bf8 [[CD]]{{\[}}[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN:        global_store_dwordx4 v{{[0-9]+}}, [[CD]]{{\[}}[[RLO]]:[[RHI]]]
define amdgpu_kernel void @test_smfmac_i32_16x16x64_fp8_bf8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.fp8.bf8(<2 x i32> %a, <4 x i32> %b, <4 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_16x16x64_fp8_fp8:
; GCN:        s_load_dwordx4 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]]{{\[}}[[RLO:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_16x16x64_fp8_fp8 [[CD]]{{\[}}[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN:        global_store_dwordx4 v{{[0-9]+}}, [[CD]]{{\[}}[[RLO]]:[[RHI]]]
define amdgpu_kernel void @test_smfmac_i32_16x16x64_fp8_fp8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <4 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.smfmac.f32.16x16x64.fp8.fp8(<2 x i32> %a, <4 x i32> %b, <4 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <4 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_32x32x32_bf8_bf8:
; GCN:        s_load_dwordx16 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]]{{\[}}[[RLO:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_32x32x32_bf8_bf8 [[CD]]{{\[}}[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]]{{\[}}[[RLO]]:{{[0-9]+}}], s[{{[0-9:]+}}]{{$}}
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:16
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:32
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9]+}}:[[RHI]]], s[{{[0-9:]+}}] offset:48
define amdgpu_kernel void @test_smfmac_i32_32x32x32_bf8_bf8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.bf8.bf8(<2 x i32> %a, <4 x i32> %b, <16 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_32x32x32_bf8_fp8:
; GCN:        s_load_dwordx16 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]]{{\[}}[[RLO:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_32x32x32_bf8_fp8 [[CD]]{{\[}}[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]]{{\[}}[[RLO]]:{{[0-9]+}}], s[{{[0-9:]+}}]{{$}}
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:16
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:32
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9]+}}:[[RHI]]], s[{{[0-9:]+}}] offset:48
define amdgpu_kernel void @test_smfmac_i32_32x32x32_bf8_fp8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.bf8.fp8(<2 x i32> %a, <4 x i32> %b, <16 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_32x32x32_fp8_bf8:
; GCN:        s_load_dwordx16 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]]{{\[}}[[RLO:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_32x32x32_fp8_bf8 [[CD]]{{\[}}[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]]{{\[}}[[RLO]]:{{[0-9]+}}], s[{{[0-9:]+}}]{{$}}
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:16
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:32
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9]+}}:[[RHI]]], s[{{[0-9:]+}}] offset:48
define amdgpu_kernel void @test_smfmac_i32_32x32x32_fp8_bf8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.fp8.bf8(<2 x i32> %a, <4 x i32> %b, <16 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

; GCN-LABEL: {{^}}test_smfmac_i32_32x32x32_fp8_fp8:
; GCN:        s_load_dwordx16 s{{\[}}[[SLO:[0-9]+]]:[[SHI:[0-9]+]]], s[{{[0-9:]+}}], 0x0{{$}}
; VGPRCD-DAG: v_mov_b64_e32 [[CD:v]]{{\[}}[[RLO:[0-9]+]]:{{[0-9]+}}], s{{\[}}[[SLO]]:{{[0-9]+}}]{{$}}
; VGPRCD-DAG: v_mov_b64_e32 v[{{[0-9]+}}:[[RHI:[0-9]+]]], s[{{[0-9]+}}:[[SHI]]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 [[CD:a]][[RLO:[0-9]+]], s[[SLO]]{{$}}
; AGPRCD-DAG: v_accvgpr_write_b32 a[[RHI:[0-9]+]], s[[SHI]]{{$}}
; GCN:        v_smfmac_f32_32x32x32_fp8_fp8 [[CD]]{{\[}}[[RLO]]:[[RHI]]], {{[av]}}[{{[0-9:]+}}], {{[av]}}[{{[0-9:]+}}], v{{[0-9]+}} cbsz:1 abid:2
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]]{{\[}}[[RLO]]:{{[0-9]+}}], s[{{[0-9:]+}}]{{$}}
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:16
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9:]+}}], s[{{[0-9:]+}}] offset:32
; GCN-DAG:    global_store_dwordx4 v{{[0-9]+}}, [[CD]][{{[0-9]+}}:[[RHI]]], s[{{[0-9:]+}}] offset:48
define amdgpu_kernel void @test_smfmac_i32_32x32x32_fp8_fp8(ptr addrspace(1) %arg, <2 x i32> %a, <4 x i32> %b, i32 %idx) #0 {
bb:
  %in.1 = load <16 x float>, ptr addrspace(1) %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.smfmac.f32.32x32x32.fp8.fp8(<2 x i32> %a, <4 x i32> %b, <16 x float> %in.1, i32 %idx, i32 1, i32 2)
  store <16 x float> %mai.1, ptr addrspace(1) %arg
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
