; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,NOLIT-SRCC,GFX908,GFX908_A %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -mattr=-mfma-inline-literal-bug -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,LIT-SRCC,GFX908,GFX908_A %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX90A,GFX908_A,GFX90A_40 %s
; RUN: llc -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX940,GFX90A_40 %s

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float, float, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float, float, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float, float, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half>, <4 x half>, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x4f16(<4 x half>, <4 x half>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half>, <4 x half>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half>, <4 x half>, <4 x float>, i32, i32, i32)
declare <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32, i32, <32 x i32>, i32, i32, i32)
declare <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32, i32, <16 x i32>, i32, i32, i32)
declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32:
; GCN-DAG:        v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:        v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN-DAG:        s_load_dwordx16
; GCN-DAG:        s_load_dwordx16
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:       v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:         v_mfma_f32_32x32x1_2b_f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908-COUNT-2: global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908-COUNT-2: global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908-COUNT-2: global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908-COUNT-2: global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-NOT:     v_accvgpr_read_b32
; GFX90A-COUNT-8: global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32:
; GCN-DAG:           v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:           v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:               s_load_dwordx16
; GFX908-COUNT-16:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-16:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:          v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:            v_mfma_f32_16x16x1_4b_f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT:      v_accvgpr_read_b32
; GFX908-COUNT-4:    global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-NOT:        v_accvgpr_read_b32
; GFX90A-COUNT-4:    global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x1f32(<16 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32:
; GCN-DAG:          v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:          v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:              s_load_dwordx4
; GFX908-COUNT-4:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-4:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:         v_mfma_f32_4x4x1f32 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:           v_mfma_f32_4x4x1_16b_f32 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-4:   v_accvgpr_read_b32
; GFX908:           global_store_dwordx4
; GFX90A-NOT:       v_accvgpr_read_b32
; GFX90A:           global_store_dwordx4 {{v[0-9]+}}, [[RES]]
define amdgpu_kernel void @test_mfma_f32_4x4x1f32(<4 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x2f32:
; GCN-DAG:           v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:           v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:               s_load_dwordx16
; GFX908-COUNT-16:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-16:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:          v_mfma_f32_32x32x2f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:            v_mfma_f32_32x32x2_f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-16:   v_accvgpr_read_b32
; GFX908-COUNT-4:    global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}]
; GFX90A-NOT:        v_accvgpr_read_b32
; GFX90A-COUNT-4:    global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x2f32(<16 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float 1.0, float 2.0, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x4f32:
; GCN-DAG:          v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:          v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN:              s_load_dwordx4
; GFX908-COUNT-4:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-4:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:         v_mfma_f32_16x16x4f32 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:           v_mfma_f32_16x16x4_f32 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-4:   v_accvgpr_read_b32
; GFX908:           global_store_dwordx4
; GFX90A-NOT:       v_accvgpr_read_b32
; GFX90A:           global_store_dwordx4 {{v[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_16x16x4f32(<4 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float 1.0, float 2.0, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x4f16:
; GCN-DAG:           s_load_dwordx16
; GCN-DAG:           s_load_dwordx16
; GFX908-COUNT-32:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-32:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:          v_mfma_f32_32x32x4f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:            v_mfma_f32_32x32x4_2b_f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-32:   v_accvgpr_read_b32
; GFX908:            global_store_dwordx4
; GFX90A-NOT:        v_accvgpr_read_b32
; GFX90A-COUNT-8:    global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_32x32x4f16(<32 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) #0 {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %c.1 = load <4 x half>, <4 x half> addrspace(1)* %c
  %c2p = getelementptr <4 x half>, <4 x half> addrspace(1)* %c, i64 1
  %c.2 = load <4 x half>, <4 x half> addrspace(1)* %c2p
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4f16(<4 x half> %c.1, <4 x half> %c.2, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x4f16:
; GCN:               s_load_dwordx16
; GFX908-COUNT-16:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-16:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:          v_mfma_f32_16x16x4f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:            v_mfma_f32_16x16x4_4b_f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-16:   v_accvgpr_read_b32
; GFX908:            global_store_dwordx4
; GFX90A-NOT:        v_accvgpr_read_b32
; GFX90A-COUNT-4:    global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_16x16x4f16(<16 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) #0 {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %c.1 = load <4 x half>, <4 x half> addrspace(1)* %c
  %c2p = getelementptr <4 x half>, <4 x half> addrspace(1)* %c, i64 1
  %c.2 = load <4 x half>, <4 x half> addrspace(1)* %c2p
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x4f16(<4 x half> %c.1, <4 x half> %c.2, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x4f16:
; GCN:              s_load_dwordx4
; GCN:              s_load_dwordx4
; GFX908-COUNT-4:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-4:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:         v_mfma_f32_4x4x4f16 [[RES:a\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:           v_mfma_f32_4x4x4_16b_f16 [[RES:a\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-4:   v_accvgpr_read_b32
; GFX908:           global_store_dwordx4
; GFX90A-NOT:       v_accvgpr_read_b32
; GFX90A:           global_store_dwordx4 {{v[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_4x4x4f16(<4 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) #0 {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %c.1 = load <4 x half>, <4 x half> addrspace(1)* %c
  %c2p = getelementptr <4 x half>, <4 x half> addrspace(1)* %c, i64 1
  %c.2 = load <4 x half>, <4 x half> addrspace(1)* %c2p
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4f16(<4 x half> %c.1, <4 x half> %c.2, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x8f16:
; GCN:               s_load_dwordx16
; GCN:               s_waitcnt lgkmcnt(0)
; GFX908_A:          v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GFX908-COUNT-16:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-16:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:          v_mfma_f32_32x32x8f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:            v_mfma_f32_32x32x8_f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-16:   v_accvgpr_read_b32
; GFX908:            global_store_dwordx4
; GFX90A-NOT:        v_accvgpr_read_b32
; GFX90A-COUNT-4:    global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_32x32x8f16(<16 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) #0 {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %c.1 = load <4 x half>, <4 x half> addrspace(1)* %c
  %c2p = getelementptr <4 x half>, <4 x half> addrspace(1)* %c, i64 1
  %c.2 = load <4 x half>, <4 x half> addrspace(1)* %c2p
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> %c.1, <4 x half> %c.2, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x16f16:
; GCN:              s_load_dwordx4
; GCN:              s_load_dwordx4
; GFX908-COUNT-4:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-4:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:         v_mfma_f32_16x16x16f16 [[RES:a\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:           v_mfma_f32_16x16x16_f16 [[RES:a\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-4:   v_accvgpr_read_b32
; GFX908:           global_store_dwordx4
; GFX90A-NOT:       v_accvgpr_read_b32
; GFX90A:           global_store_dwordx4 {{v[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_16x16x16f16(<4 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) #0 {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %c.1 = load <4 x half>, <4 x half> addrspace(1)* %c
  %c2p = getelementptr <4 x half>, <4 x half> addrspace(1)* %c, i64 1
  %c.2 = load <4 x half>, <4 x half> addrspace(1)* %c2p
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16f16(<4 x half> %c.1, <4 x half> %c.2, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_32x32x4i8:
; GCN-DAG:         v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG:         v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN-DAG:         s_load_dwordx16
; GCN-DAG:         s_load_dwordx16
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-32: v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:        v_mfma_i32_32x32x4i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:          v_mfma_i32_32x32x4_2b_i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-8:  global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_i32_32x32x4i8(<32 x i32> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <32 x i32>, <32 x i32> addrspace(1)* %arg
  %mai.1 = tail call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 1, i32 2, <32 x i32> %in.1, i32 1, i32 2, i32 3)
  store <32 x i32> %mai.1, <32 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_16x16x4i8:
; GCN-DAG:           v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG:           v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN:               s_load_dwordx16
; GFX908-COUNT-16:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-16:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:          v_mfma_i32_16x16x4i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:            v_mfma_i32_16x16x4_4b_i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-16:   v_accvgpr_read_b32
; GFX908:            global_store_dwordx4
; GFX90A-NOT:        v_accvgpr_read_b32
; GFX90A-COUNT-4:    global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_i32_16x16x4i8(<16 x i32> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <16 x i32>, <16 x i32> addrspace(1)* %arg
  %mai.1 = tail call <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32 1, i32 2, <16 x i32> %in.1, i32 1, i32 2, i32 3)
  store <16 x i32> %mai.1, <16 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_4x4x4i8:
; GCN-DAG:          v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG:          v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN:              s_load_dwordx4
; GFX908-COUNT-4:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-COUNT-4:v_accvgpr_write_b32 a{{[0-9]+}}, s{{[0-9]+}}
; GFX908_A:         v_mfma_i32_4x4x4i8 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:           v_mfma_i32_4x4x4_16b_i8 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-4:   v_accvgpr_read_b32
; GFX908:           global_store_dwordx4
; GFX90A-NOT:       v_accvgpr_read_b32
; GFX90A:           global_store_dwordx4 {{v[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_i32_4x4x4i8(<4 x i32> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <4 x i32>, <4 x i32> addrspace(1)* %arg
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %in.1, i32 1, i32 2, i32 3)
  store <4 x i32> %mai.1, <4 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_forward_acc:
; GFX908_A:      v_mfma_f32_32x32x1f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX908_A-NEXT: v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
; GFX940:        v_mfma_f32_32x32x1_2b_f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX940-NEXT:   v_mfma_f32_32x32x1_2b_f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_forward_acc(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %mai.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.2, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32_forward_acc:
; GFX908_A:      v_mfma_f32_16x16x1f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX908_A-NEXT: v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
; GFX940:        v_mfma_f32_16x16x1_4b_f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX940-NEXT:   v_mfma_f32_16x16x1_4b_f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
define amdgpu_kernel void @test_mfma_f32_16x16x1f32_forward_acc(<16 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %mai.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.2, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_forward_acc:
; GFX908_A:      v_mfma_f32_4x4x1f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX908_A-NEXT: v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
; GFX940:        v_mfma_f32_4x4x1_16b_f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX940-NEXT:   s_nop 1
; GFX940-NEXT:   v_mfma_f32_4x4x1_16b_f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_forward_acc(<4 x float> addrspace(1)* %arg) #0 {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %mai.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.2, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_imm_splat:
; GCN-DAG:        v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:        v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC:     v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9:]+}}]
; LIT-SRCC:       v_mfma_f32_4x4x1f32 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], 1.0
; GFX90A:         v_mfma_f32_4x4x1f32 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], 1.0
; GFX940:         v_mfma_f32_4x4x1_16b_f32 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], 1.0
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908:         global_store_dwordx4
; GFX90A-NOT:     v_accvgpr_read_b32
; GFX90A:         global_store_dwordx4 {{v[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_imm_splat(<4 x float> addrspace(1)* %arg) #0 {
bb:
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32_imm_splat:
; GCN-DAG:         v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:         v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; NOLIT-SRCC-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC:      v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9:]+}}]
; LIT-SRCC:        v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 1.0
; GFX90A:          v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 1.0
; GFX940:          v_mfma_f32_16x16x1_4b_f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 1.0
; GFX908-COUNT-16: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-4:  global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_16x16x1f32_imm_splat(<16 x float> addrspace(1)* %arg) #0 {
bb:
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x8f16_imm_splat:
; GCN-DAG:         v_mov_b32_e32 v[[TWO:[0-9]+]], 0x40004000
; GCN-DAG:         v_mov_b32_e32 v[[ONE:[0-9]+]], 0x3c003c00
; NOLIT-SRCC-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC:      v_mfma_f32_32x32x8f16 a[{{[0-9]+:[0-9]+}}], v[[[ONE]]:{{[0-9]+}}], v[[[TWO]]:{{[0-9]+}}], a[{{[0-9:]+}}]
; LIT-SRCC:        v_mfma_f32_32x32x8f16 a[{{[0-9]+:[0-9]+}}], v[[[ONE]]:{{[0-9]+}}], v[[[TWO]]:{{[0-9]+}}], 1.0
; GFX90A:          v_mfma_f32_32x32x8f16 a[{{[0-9]+:[0-9]+}}], v[[[ONE]]:{{[0-9]+}}], v[[[TWO]]:{{[0-9]+}}], 1.0
; GFX940:          v_mfma_f32_32x32x8_f16 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[ONE]]:{{[0-9]+}}], v{{\[}}[[TWO]]:{{[0-9]+}}], 1.0
; GFX908-COUNT-16: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-4:  global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_32x32x8f16_imm_splat(<16 x float> addrspace(1)* %arg) #0 {
bb:
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>, <4 x half> <half 2.0, half 2.0, half 2.0, half 2.0>, <16 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_imm_splat:
; GCN-DAG:         v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:         v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; NOLIT-SRCC-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, 0
; NOLIT-SRCC:      v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9:]+}}]
; LIT-SRCC:        v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 0
; GFX90A:          v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 0
; GFX940:          v_mfma_f32_32x32x1_2b_f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 0
; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-8:  global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_imm_splat(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> <float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_imm:
; GCN-DAG:        v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG:        v_accvgpr_write_b32 a{{[0-9]+}}, 2.0
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GFX908-DAG:     v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GFX90A-DAG:     v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:     v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX908_A:       v_mfma_f32_4x4x1f32 [[RES:a\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX940:         v_mfma_f32_4x4x1_16b_f32 [[RES:a\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908:         global_store_dwordx4
; GFX90A-NOT:     v_accvgpr_read_b32
; GFX90A:         global_store_dwordx4 {{v[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_imm(<4 x float> addrspace(1)* %arg) #0 {
bb:
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> <float 1.0, float 2.0, float 1.0, float 1.0>, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32_imm:
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, 2.0
; GFX908-COUNT-14: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GFX90A-COUNT-14: v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX908_A:        v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX940:          v_mfma_f32_16x16x1_4b_f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX908-COUNT-16: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-4:  global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_16x16x1f32_imm(<16 x float> addrspace(1)* %arg) #0 {
bb:
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 2.0>, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_imm:
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX908-DAG:      v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX90A-DAG:      v_accvgpr_mov_b32 a{{[0-9]+}}, a{{[0-9]+}}
; GFX908_A:        v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX940:          v_mfma_f32_32x32x1_2b_f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-8:  global_store_dwordx4 {{v[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_imm(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> <float 1.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_lit_splat:
; GFX908:         v_mov_b32_e32 [[TMP:v[0-9]+]], 0x42f60000
; GFX90A_40:      s_mov_b32 [[TMP:s[0-9]+]], 0x42f60000
; GCN:            v_accvgpr_write_b32 [[TTMPA:a[0-9]+]], [[TMP]]
; GFX908:         v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GFX908:         v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GFX908:         v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GFX90A:         v_accvgpr_mov_b32 a{{[0-9]+}}, [[TTMPA]]
; GFX90A:         v_accvgpr_mov_b32 a{{[0-9]+}}, [[TTMPA]]
; GFX90A:         v_accvgpr_mov_b32 a{{[0-9]+}}, [[TTMPA]]
; GFX908_A:       v_mfma_f32_4x4x1f32 [[RES:a\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX940:         v_mfma_f32_4x4x1_16b_f32 [[RES:a\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908:         global_store_dwordx4
; GFX90A-NOT:     v_accvgpr_read_b32
; GFX90A:         global_store_dwordx4 {{v[0-9]+}}, [[RES]]
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_lit_splat(<4 x float> addrspace(1)* %arg, i64 %idx) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %arg, i32 %tid
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> <float 123.0, float 123.0, float 123.0, float 123.0>, i32 0, i32 0, i32 0)
  ;store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %gep
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_lit_splat_bad_code:
; GFX908:   v_mov_b32_e32 [[TMP0:v[0-9]+]], 0x42f60000
; GFX90A_40:s_mov_b32 [[TMP0:s[0-9]+]], 0x42f60000
; GCN:      v_accvgpr_write_b32 [[AGPR:a[0-9]+]], [[TMP0]]
; GFX90A_40-COUNT-3: v_accvgpr_mov_b32 a{{[0-9]+}}, [[AGPR]]
; GFX908-NEXT:   v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP0]]
; GFX908-NEXT:   v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP0]]
; GFX908-NEXT:   v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP0]]
; GCN: s_nop 0
; GFX908_A:  v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX940:    v_mfma_f32_4x4x1_16b_f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908:    global_store_dwordx4 v{{[0-9]+}}, v[{{[0-9:]+}}], s[{{[0-9:]+}}]
; GFX90A_40: global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}], s[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_lit_splat_bad_code(<4 x float> addrspace(1)* %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <4 x float>, <4 x float> addrspace(1)* %arg, i32 %tid

  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> <float 123.0, float 123.0, float 123.0, float 123.0>, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_vecarg:
; GFX90A_40-DAG:     v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GFX90A_40-DAG:     v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN-COUNT-8:       global_load_dwordx4
; GFX908-COUNT-16:   v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A_40-NOT:     v_accvgpr_write
; GFX908-DAG:        v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GFX908-DAG:        v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GFX908:            v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX90A:            v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX940:            v_mfma_f32_32x32x1_2b_f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908:            v_accvgpr_read_b32
; GFX908-COUNT-8:    global_store_dwordx4
; GFX90A_40-NOT:     v_accvgpr_read_b32
; GFX90A_40-COUNT-5: global_store_dwordx4 v{{[0-9:]+}}, a[{{[0-9:]+}}], s[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_vecarg(<32 x float> addrspace(1)* %arg) #0 {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %arg, i32 %tid
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %gep
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %gep
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,256" }
