; RUN: llc -global-isel=0 -march=amdgcn -mcpu=tahiti -verify-machineinstrs -stop-after=amdgpu-isel -o - %s | FileCheck %s --check-prefixes=GCN,PREGFX12-SDAG
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=tahiti -verify-machineinstrs -stop-after=instruction-select -o - %s | FileCheck %s --check-prefixes=GCN,PREGFX12-GISEL
; RUN: llc -global-isel=0 -march=amdgcn -mcpu=gfx1200 -verify-machineinstrs -stop-after=amdgpu-isel -o - %s | FileCheck %s --check-prefixes=GCN,GFX12PLUS-SDAG
; RUN: llc -global-isel=1 -march=amdgcn -mcpu=gfx1200 -verify-machineinstrs -stop-after=instruction-select -o - %s | FileCheck %s --check-prefixes=GCN,GFX12PLUS-GISEL

; GCN-LABEL: name: buffer_swizzle_bit_pregfx12
; PREGFX12-SDAG: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_IDXEN {{%[0-9]+}}, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, 0, 1, implicit $exec
; PREGFX12-GISEL: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_IDXEN {{%[0-9]+}}, {{%[0-9]+}}, {{%[0-9]+}}, 0, 0, 1, implicit $exec
; GFX12PLUS-SDAG: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_VBUFFER_IDXEN killed {{%[0-9]+}}, killed {{%[0-9]+}}, $sgpr_null, 0, 8, 0, implicit $exec
; GFX12PLUS-GISEL: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_VBUFFER_IDXEN {{%[0-9]+}}, {{%[0-9]+}}, $sgpr_null, 0, 8, 0, implicit $exec
define amdgpu_ps <4 x float> @buffer_swizzle_bit_pregfx12(<4 x i32> inreg %0) {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 8)
  ret <4 x float> %data
}

; GCN-LABEL: name: buffer_swizzle_bit_gfx12plus
; PREGFX12-SDAG: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_IDXEN {{%[0-9]+}}, killed {{%[0-9]+}}, {{%[0-9]+}}, 0, 0, 0, implicit $exec
; PREGFX12-GISEL: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_IDXEN {{%[0-9]+}}, {{%[0-9]+}}, {{%[0-9]+}}, 0, 0, 0, implicit $exec
; GFX12PLUS-SDAG: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_VBUFFER_IDXEN killed {{%[0-9]+}}, killed {{%[0-9]+}}, $sgpr_null, 0, 0, 1, implicit $exec
; GFX12PLUS-GISEL: {{%[0-9]+}}:vreg_128 = BUFFER_LOAD_DWORDX4_VBUFFER_IDXEN {{%[0-9]+}}, {{%[0-9]+}}, $sgpr_null, 0, 0, 1, implicit $exec
define amdgpu_ps <4 x float> @buffer_swizzle_bit_gfx12plus(<4 x i32> inreg %0) {
  %data = call <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32> %0, i32 0, i32 0, i32 0, i32 64)
  ret <4 x float> %data
}

declare <4 x float> @llvm.amdgcn.struct.buffer.load.v4f32(<4 x i32>, i32, i32, i32, i32)
