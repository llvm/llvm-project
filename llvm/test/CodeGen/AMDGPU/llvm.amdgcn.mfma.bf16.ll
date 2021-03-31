; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -mattr=-mfma-inline-literal-bug -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX90A %s

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16>, <2 x i16>, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16>, <2 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16>, <2 x i16>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16>, <2 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16>, <2 x i16>, <4 x float>, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}test_mfma_f32_32x32x2bf16:
; GCN-DAG:         v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG:         v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN-DAG:         s_load_dwordx16
; GCN-DAG:         s_load_dwordx16
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG:         v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN:             v_mfma_f32_32x32x2bf16 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-32: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-8:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_32x32x2bf16(<32 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %a = bitcast i32 1 to <2 x i16>
  %b = bitcast i32 2 to <2 x i16>
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16> %a, <2 x i16> %b, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x2bf16:
; GCN-DAG:         v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG:         v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN:             s_load_dwordx16
; GCN-COUNT-16:    v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN:             v_mfma_f32_16x16x2bf16 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-16: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-4:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_16x16x2bf16(<16 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %a = bitcast i32 1 to <2 x i16>
  %b = bitcast i32 2 to <2 x i16>
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16> %a, <2 x i16> %b, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x2bf16:
; GCN-DAG:        v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG:        v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN:            s_load_dwordx4
; GCN-COUNT-4:    v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN:            v_mfma_f32_4x4x2bf16 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908:         global_store_dwordx4
; GFX90A-NOT:     v_accvgpr_read_b32
; GFX90A:         global_store_dwordx4 v{{[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_4x4x2bf16(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %a = bitcast i32 1 to <2 x i16>
  %b = bitcast i32 2 to <2 x i16>
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x4bf16:
; GCN-DAG:         v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG:         v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN:             s_load_dwordx16
; GCN-COUNT-16:    v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN:             v_mfma_f32_32x32x4bf16 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-16: v_accvgpr_read_b32
; GFX908:          global_store_dwordx4
; GFX90A-NOT:      v_accvgpr_read_b32
; GFX90A-COUNT-4:  global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}],
define amdgpu_kernel void @test_mfma_f32_32x32x4bf16(<16 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %a = bitcast i32 1 to <2 x i16>
  %b = bitcast i32 2 to <2 x i16>
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16> %a, <2 x i16> %b, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x8bf16:
; GCN-DAG:        v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG:        v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN:            s_load_dwordx4
; GCN-COUNT-4:    v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN:            v_mfma_f32_16x16x8bf16 [[RES:a\[[0-9]+:[0-9]+\]]], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GFX908-COUNT-4: v_accvgpr_read_b32
; GFX908:         global_store_dwordx4
; GFX90A-NOT:     v_accvgpr_read_b32
; GFX90A:         global_store_dwordx4 v{{[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_16x16x8bf16(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %a = bitcast i32 1 to <2 x i16>
  %b = bitcast i32 2 to <2 x i16>
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}
