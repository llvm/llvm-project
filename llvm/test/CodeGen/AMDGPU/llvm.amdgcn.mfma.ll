; RUN: llc -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,NOLIT-SRCC %s
; RUN: llc -march=amdgcn -mcpu=gfx908 -mattr=-mfma-inline-literal-bug -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,LIT-SRCC %s

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
declare <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32, i32, <16 x i32>, i32, i32, i32)
declare <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32, i32, <4 x i32>, i32, i32, i32)
declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x2bf16(<2 x i16>, <2 x i16>, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x2bf16(<2 x i16>, <2 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x2bf16(<2 x i16>, <2 x i16>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16(<2 x i16>, <2 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16>, <2 x i16>, <4 x float>, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN-DAG: s_load_dwordx16
; GCN-DAG: s_load_dwordx16
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_32x32x1f32(<32 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN: s_load_dwordx16
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_16x16x1f32(<16 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN: s_load_dwordx4
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_4x4x1f32(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x2f32:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN: s_load_dwordx16
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_32x32x2f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_32x32x2f32(<16 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x2f32(float 1.0, float 2.0, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x4f32:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN: s_load_dwordx4
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_16x16x4f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_16x16x4f32(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float 1.0, float 2.0, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x4f16:
; GCN-DAG: s_load_dwordx16
; GCN-DAG: s_load_dwordx16
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_32x32x4f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_32x32x4f16(<32 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) {
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
; GCN: s_load_dwordx16
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_16x16x4f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_16x16x4f16(<16 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) {
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
; GCN: s_load_dwordx4
; GCN: s_load_dwordx2
; GCN: s_load_dwordx2
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_4x4x4f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_4x4x4f16(<4 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) {
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
; GCN: s_load_dwordx16
; GCN: s_waitcnt lgkmcnt(0)
; GCN: v_mov_b32_e32 v{{[0-9]+}}, s{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_32x32x8f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_32x32x8f16(<16 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) {
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
; GCN: s_load_dwordx4
; GCN: s_load_dwordx4
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_16x16x16f16 a[{{[0-9]+:[0-9]+}}], {{v\[[0-9]+:[0-9]+\]}}, {{v\[[0-9]+:[0-9]+\]}}, a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_16x16x16f16(<4 x float> addrspace(1)* %arg, <4 x half> addrspace(1)* %c) {
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
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN-DAG: s_load_dwordx16
; GCN-DAG: s_load_dwordx16
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_i32_32x32x4i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_i32_32x32x4i8(<32 x i32> addrspace(1)* %arg) {
bb:
  %in.1 = load <32 x i32>, <32 x i32> addrspace(1)* %arg
  %mai.1 = tail call <32 x i32> @llvm.amdgcn.mfma.i32.32x32x4i8(i32 1, i32 2, <32 x i32> %in.1, i32 1, i32 2, i32 3)
  store <32 x i32> %mai.1, <32 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_16x16x4i8:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: s_load_dwordx16
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_i32_16x16x4i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_i32_16x16x4i8(<16 x i32> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x i32>, <16 x i32> addrspace(1)* %arg
  %mai.1 = tail call <16 x i32> @llvm.amdgcn.mfma.i32.16x16x4i8(i32 1, i32 2, <16 x i32> %in.1, i32 1, i32 2, i32 3)
  store <16 x i32> %mai.1, <16 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_4x4x4i8:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: s_load_dwordx4
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_i32_4x4x4i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: global_store_dwordx4
define amdgpu_kernel void @test_mfma_i32_4x4x4i8(<4 x i32> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x i32>, <4 x i32> addrspace(1)* %arg
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %in.1, i32 1, i32 2, i32 3)
  store <4 x i32> %mai.1, <4 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_32x32x8i8:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: s_load_dwordx16
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_i32_32x32x8i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_i32_32x32x8i8(<16 x i32> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x i32>, <16 x i32> addrspace(1)* %arg
  %mai.1 = tail call <16 x i32> @llvm.amdgcn.mfma.i32.32x32x8i8(i32 1, i32 2, <16 x i32> %in.1, i32 1, i32 2, i32 3)
  store <16 x i32> %mai.1, <16 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_i32_16x16x16i8:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: s_load_dwordx4
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_i32_16x16x16i8 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_i32_16x16x16i8(<4 x i32> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x i32>, <4 x i32> addrspace(1)* %arg
  %mai.1 = tail call <4 x i32> @llvm.amdgcn.mfma.i32.16x16x16i8(i32 1, i32 2, <4 x i32> %in.1, i32 1, i32 2, i32 3)
  store <4 x i32> %mai.1, <4 x i32> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x2bf16:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN-DAG: s_load_dwordx16
; GCN-DAG: s_load_dwordx16
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_32x32x2bf16 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
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
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: s_load_dwordx16
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_16x16x2bf16 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
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
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: s_load_dwordx4
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_4x4x2bf16 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
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
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: s_load_dwordx16
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_32x32x4bf16 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
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
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1
; GCN: s_load_dwordx4
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mfma_f32_16x16x8bf16 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_16x16x8bf16(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %a = bitcast i32 1 to <2 x i16>
  %b = bitcast i32 2 to <2 x i16>
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x8bf16(<2 x i16> %a, <2 x i16> %b, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_forward_acc:
; GCN:      v_mfma_f32_32x32x1f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GCN-NEXT: v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_forward_acc(<32 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %mai.1, i32 0, i32 0, i32 0)
  store <32 x float> %mai.2, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32_forward_acc:
; GCN:      v_mfma_f32_16x16x1f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GCN-NEXT: v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
define amdgpu_kernel void @test_mfma_f32_16x16x1f32_forward_acc(<16 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> %mai.1, i32 0, i32 0, i32 0)
  store <16 x float> %mai.2, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_forward_acc:
; GCN:      v_mfma_f32_4x4x1f32 [[MAI1:a\[[0-9]+:[0-9]+\]]], v{{[0-9]+}}, v{{[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GCN-NEXT: v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], v{{[0-9]+}}, v{{[0-9]+}}, [[MAI1]]
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_forward_acc(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %in.1, i32 0, i32 0, i32 0)
  %mai.2 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> %mai.1, i32 0, i32 0, i32 0)
  store <4 x float> %mai.2, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_imm_splat:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC: v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9:]+}}]
; LIT-SRCC: v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 1.0
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_imm_splat(<4 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32_imm_splat:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC: v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9:]+}}]
; LIT-SRCC: v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 1.0
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_16x16x1f32_imm_splat(<16 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x8f16_imm_splat:
; GCN-DAG: v_mov_b32_e32 v[[TWO:[0-9]+]], 0x40004000
; GCN-DAG: v_mov_b32_e32 v[[ONE:[0-9]+]], 0x3c003c00
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; NOLIT-SRCC: v_mfma_f32_32x32x8f16 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[ONE]]:{{[0-9]+}}], v{{\[}}[[TWO]]:{{[0-9]+}}], a[{{[0-9:]+}}]
; LIT-SRCC: v_mfma_f32_32x32x8f16 a[{{[0-9]+:[0-9]+}}], v{{\[}}[[ONE]]:{{[0-9]+}}], v{{\[}}[[TWO]]:{{[0-9]+}}], 1.0
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_32x32x8f16_imm_splat(<16 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half> <half 1.0, half 1.0, half 1.0, half 1.0>, <4 x half> <half 2.0, half 2.0, half 2.0, half 2.0>, <16 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0>, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_imm_splat:
; GCN-DAG: v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG: v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; NOLIT-SRCC-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; NOLIT-SRCC: v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9:]+}}]
; LIT-SRCC: v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], 0
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_imm_splat(<32 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> <float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_imm:
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 2.0
; GCN: v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_imm(<4 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> <float 1.0, float 2.0, float 1.0, float 1.0>, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x1f32_imm:
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 2.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN: v_mfma_f32_16x16x1f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_16x16x1f32_imm(<16 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float 1.0, float 2.0, <16 x float> <float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 1.0, float 2.0>, i32 0, i32 0, i32 0)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_imm:
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 1.0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN-DAG: v_accvgpr_write_b32 a{{[0-9]+}}, 0
; GCN: v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: v_accvgpr_read_b32
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
; GCN-DAG: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_imm(<32 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> <float 1.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0, float 0.0>, i32 0, i32 0, i32 0)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x1f32_lit_splat:
; GCN: v_mov_b32_e32 [[TMP:v[0-9]+]], 0x42f60000
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GCN: v_accvgpr_write_b32 a{{[0-9]+}}, [[TMP]]
; GCN: v_mfma_f32_4x4x1f32 a[{{[0-9]+:[0-9]+}}], {{v[0-9]+}}, {{v[0-9]+}}, a[{{[0-9]+:[0-9]+}}]
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: v_accvgpr_read_b32
; GCN: global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_4x4x1f32_lit_splat(<4 x float> addrspace(1)* %arg) {
bb:
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x1f32(float 1.0, float 2.0, <4 x float> <float 123.0, float 123.0, float 123.0, float 123.0>, i32 0, i32 0, i32 0)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x1f32_vecarg:
; GCN-DAG:         v_mov_b32_e32 [[TWO:v[0-9]+]], 2.0
; GCN-DAG:         v_mov_b32_e32 [[ONE:v[0-9]+]], 1.0
; GCN-COUNT-8:     global_load_dwordx4
; GCN-COUNT-16:    v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GCN:             v_mfma_f32_32x32x1f32 a[{{[0-9]+:[0-9]+}}], [[ONE]], [[TWO]], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-COUNT-32:    v_accvgpr_read_b32
; GCN-COUNT-8:     global_store_dwordx4
define amdgpu_kernel void @test_mfma_f32_32x32x1f32_vecarg(<32 x float> addrspace(1)* %arg) {
bb:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds <32 x float>, <32 x float> addrspace(1)* %arg, i32 %tid
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %gep
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x1f32(float 1.0, float 2.0, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %gep
  ret void
}
