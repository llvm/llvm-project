; RUN: llc -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -enable-var-scope --check-prefixes=GCN,GFX90A %s

declare <32 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16.1k(<4 x i16>, <4 x i16>, <32 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x4bf16.1k(<4 x i16>, <4 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.4x4x4bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32, i32, i32)
declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8bf16.1k(<4 x i16>, <4 x i16>, <16 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32, i32, i32)
declare <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double, double, <4 x double>, i32, i32, i32)
declare double @llvm.amdgcn.mfma.f64.4x4x4f64(double, double, double, i32, i32, i32)
declare i32 @llvm.amdgcn.workitem.id.x()

; GCN-LABEL: {{^}}test_mfma_f32_32x32x4bf16_1k:
; GCN-DAG:     s_load_dwordx16
; GCN-DAG:     s_load_dwordx16
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A-DAG:  v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GFX90A-DAG:  v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX90A:      v_mfma_f32_32x32x4bf16_1k a[{{[0-9]+:[0-9]+}}], v{{\[}}[[ONE]]:{{[0-9+]}}], v{{\[}}[[TWO]]:{{[0-9+]}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:     v_accvgpr_read_b32
; GCN-COUNT-8: global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x4bf16_1k(<32 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <32 x float>, <32 x float> addrspace(1)* %arg
  %a = bitcast i64 1 to <4 x i16>
  %b = bitcast i64 2 to <4 x i16>
  %mai.1 = tail call <32 x float> @llvm.amdgcn.mfma.f32.32x32x4bf16.1k(<4 x i16> %a, <4 x i16> %b, <32 x float> %in.1, i32 1, i32 2, i32 3)
  store <32 x float> %mai.1, <32 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x4bf16_1k:
; GCN-DAG:         s_load_dwordx16
; GCN-DAG:         v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GCN-DAG:         v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX90A-COUNT-16: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A:          v_mfma_f32_16x16x4bf16_1k a[{{[0-9]+:[0-9]+}}], v{{\[}}[[ONE]]:{{[0-9+]}}], v{{\[}}[[TWO]]:{{[0-9+]}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:         v_accvgpr_read_b32
; GCN-COUNT-4:     global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_16x16x4bf16_1k(<16 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %a = bitcast i64 1 to <4 x i16>
  %b = bitcast i64 2 to <4 x i16>
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.16x16x4bf16.1k(<4 x i16> %a, <4 x i16> %b, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_4x4x4bf16_1k:
; GCN-DAG:        s_load_dwordx4
; GCN-DAG:        v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GCN-DAG:        v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX90A-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A:         v_mfma_f32_4x4x4bf16_1k [[RES:a\[[0-9]+:[0-9]+\]]], v{{\[}}[[ONE]]:{{[0-9+]}}], v{{\[}}[[TWO]]:{{[0-9+]}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:        v_accvgpr_read_b32
; GCN:            global_store_dwordx4 v{{[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_4x4x4bf16_1k(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %a = bitcast i64 1 to <4 x i16>
  %b = bitcast i64 2 to <4 x i16>
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.4x4x4bf16.1k(<4 x i16> %a, <4 x i16> %b, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_32x32x8bf16_1k:
; GCN-DAG:         s_load_dwordx16
; GCN-DAG:         v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GCN-DAG:         v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX90A-COUNT-16: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A:          v_mfma_f32_32x32x8bf16_1k a[{{[0-9]+:[0-9]+}}], v{{\[}}[[ONE]]:{{[0-9+]}}], v{{\[}}[[TWO]]:{{[0-9+]}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:         v_accvgpr_read_b32
; GCN-COUNT-4:     global_store_dwordx4 v{{[0-9]+}}, a[{{[0-9:]+}}]
define amdgpu_kernel void @test_mfma_f32_32x32x8bf16_1k(<16 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <16 x float>, <16 x float> addrspace(1)* %arg
  %a = bitcast i64 1 to <4 x i16>
  %b = bitcast i64 2 to <4 x i16>
  %mai.1 = tail call <16 x float> @llvm.amdgcn.mfma.f32.32x32x8bf16.1k(<4 x i16> %a, <4 x i16> %b, <16 x float> %in.1, i32 1, i32 2, i32 3)
  store <16 x float> %mai.1, <16 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f32_16x16x16bf16_1k:
; GCN-DAG:        s_load_dwordx4
; GCN-DAG:        v_mov_b32_e32 v[[TWO:[0-9]+]], 2
; GCN-DAG:        v_mov_b32_e32 v[[ONE:[0-9]+]], 1
; GFX90A-COUNT-4: v_accvgpr_write_b32 a{{[0-9]+}}, v{{[0-9]+}}
; GFX90A:         v_mfma_f32_16x16x16bf16_1k [[RES:a\[[0-9]+:[0-9]+\]]], v{{\[}}[[ONE]]:{{[0-9+]}}], v{{\[}}[[TWO]]:{{[0-9+]}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN-NOT:        v_accvgpr_read_b32
; GCN:            global_store_dwordx4 v{{[0-9]+}}, [[RES]],
define amdgpu_kernel void @test_mfma_f32_16x16x16bf16_1k(<4 x float> addrspace(1)* %arg) {
bb:
  %in.1 = load <4 x float>, <4 x float> addrspace(1)* %arg
  %a = bitcast i64 1 to <4 x i16>
  %b = bitcast i64 2 to <4 x i16>
  %mai.1 = tail call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %a, <4 x i16> %b, <4 x float> %in.1, i32 1, i32 2, i32 3)
  store <4 x float> %mai.1, <4 x float> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f64_4x4x4f64:
; GFX90A: v_mfma_f64_4x4x4f64 [[M1:a\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], 0{{$}}
; GFX90A: v_mfma_f64_4x4x4f64 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], [[M1]] cbsz:1 abid:2 blgp:3
; GCN:    global_store_dwordx2
define amdgpu_kernel void @test_mfma_f64_4x4x4f64(double addrspace(1)* %arg, double %a, double %b) {
bb:
  %mai.1 = tail call double @llvm.amdgcn.mfma.f64.4x4x4f64(double %a, double %b, double 0.0, i32 0, i32 0, i32 0)
  %mai.2 = tail call double @llvm.amdgcn.mfma.f64.4x4x4f64(double %a, double %b, double %mai.1, i32 1, i32 2, i32 3)
  store double %mai.2, double addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f64_16x16x4f64:
; GCN:    s_load_dwordx8
; GFX90A: v_mfma_f64_16x16x4f64 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}] cbsz:1 abid:2 blgp:3
; GCN:    global_store_dwordx4
; GCN:    global_store_dwordx4
define amdgpu_kernel void @test_mfma_f64_16x16x4f64(<4 x double> addrspace(1)* %arg, double %a, double %b) {
bb:
  %in.1 = load <4 x double>, <4 x double> addrspace(1)* %arg
  %mai.1 = tail call <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double %a, double %b, <4 x double> %in.1, i32 1, i32 2, i32 3)
  store <4 x double> %mai.1, <4 x double> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f64_16x16x4f64_splat_imm:
; GFX90A: v_mfma_f64_16x16x4f64 [[M1:a\[[0-9]+:[0-9]+\]]], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], 0{{$}}
; GFX90A: v_mfma_f64_16x16x4f64 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], [[M1]] cbsz:1 abid:2 blgp:3
; GCN:    global_store_dwordx4
; GCN:    global_store_dwordx4
define amdgpu_kernel void @test_mfma_f64_16x16x4f64_splat_imm(<4 x double> addrspace(1)* %arg, double %a, double %b) {
bb:
  %mai.1 = tail call <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double %a, double %b, <4 x double> <double 0.0, double 0.0, double 0.0, double 0.0>, i32 0, i32 0, i32 0)
  %mai.2 = tail call <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double %a, double %b, <4 x double> %mai.1, i32 1, i32 2, i32 3)
  store <4 x double> %mai.2, <4 x double> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f64_16x16x4f64_imm:
; GFX90A: v_mfma_f64_16x16x4f64 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}]{{$}}
; GCN:    global_store_dwordx4
; GCN:    global_store_dwordx4
define amdgpu_kernel void @test_mfma_f64_16x16x4f64_imm(<4 x double> addrspace(1)* %arg, double %a, double %b) {
bb:
  %mai.1 = tail call <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double %a, double %b, <4 x double> <double 0.0, double 0.0, double 0.0, double 1.0>, i32 0, i32 0, i32 0)
  store <4 x double> %mai.1, <4 x double> addrspace(1)* %arg
  ret void
}

; GCN-LABEL: {{^}}test_mfma_f64_16x16x4f64_splat_lit:
; GCN-DAG:    v_accvgpr_write_b32 a{{[0-9]+}}, 0{{$}}
; GFX90A-DAG: v_mov_b32_e32 v{{[0-9]+}}, 0x405ec000
; GFX90A:     v_mfma_f64_16x16x4f64 a[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], v[{{[0-9]+:[0-9]+}}], a[{{[0-9]+:[0-9]+}}]{{$}}
; GCN:        global_store_dwordx4
; GCN:        global_store_dwordx4
define amdgpu_kernel void @test_mfma_f64_16x16x4f64_splat_lit(<4 x double> addrspace(1)* %arg, double %a, double %b) {
bb:
  %mai.1 = tail call <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double %a, double %b, <4 x double> <double 123.0, double 123.0, double 123.0, double 123.0>, i32 0, i32 0, i32 0)
  store <4 x double> %mai.1, <4 x double> addrspace(1)* %arg
  ret void
}
