; RUN: llc -O0 -mtriple=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX9 %s
; RUN: llc -O0 -mtriple=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX9 %s
; RUN: llc -O0 -mtriple=amdgcn -mcpu=gfx9-generic --amdhsa-code-object-version=6 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX9 %s
; RUN: not llc -O0 -mtriple=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefixes=GFX90A %s
; RUN: not llc -O0 -mtriple=amdgcn -mcpu=gfx942 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefixes=GFX942 %s
; RUN: llc -O0 -mtriple=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX1030 %s
; RUN: llc -O0 -mtriple=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX1100 %s

; GFX9-LABEL: image_sample_test:
; GFX9: image_sample_lz

; GFX90A: error: <unknown>:0:0: in function image_sample_test void (ptr addrspace(1), float, float, <8 x i32>, <4 x i32>): requested image instruction is not supported on this GPU
; GFX942: error: <unknown>:0:0: in function image_sample_test void (ptr addrspace(1), float, float, <8 x i32>, <4 x i32>): requested image instruction is not supported on this GPU

; GFX1030-LABEL: image_sample_test:
; GFX1030: image_sample_lz

; GFX1100-LABEL: image_sample_test:
; GFX1100: image_sample_lz

define amdgpu_kernel void @image_sample_test(ptr addrspace(1) %out, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) {

  %result = tail call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0)

  store <4 x float> %result, ptr addrspace(1) %out
  ret void
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)

; GFX90A: error: <unknown>:0:0: in function sample_1d_tfe <4 x float> (<8 x i32>, <4 x i32>, ptr addrspace(1), float): TFE is not supported on this GPU
; GFX942: error: <unknown>:0:0: in function sample_1d_tfe <4 x float> (<8 x i32>, <4 x i32>, ptr addrspace(1), float): TFE is not supported on this GPU
define <4 x float> @sample_1d_tfe(<8 x i32> inreg %rsrc, <4 x i32> inreg %samp, ptr addrspace(1) inreg %out, float %s) {
  %v = call {<4 x float>,i32} @llvm.amdgcn.image.sample.1d.v4f32i32.f32(i32 15, float %s, <8 x i32> %rsrc, <4 x i32> %samp, i1 0, i32 1, i32 0)
  %v.vec = extractvalue {<4 x float>, i32} %v, 0
  %v.err = extractvalue {<4 x float>, i32} %v, 1
  store i32 %v.err, ptr addrspace(1) %out, align 4
  ret <4 x float> %v.vec
}
