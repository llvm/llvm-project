; RUN: llc -O0 -march=amdgcn -mcpu=gfx906 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX906 %s
; RUN: llc -O0 -march=amdgcn -mcpu=gfx908 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX908 %s
; RUN: llc -O0 -march=amdgcn -mcpu=gfx90a -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX90A %s
; RUN: llc -O0 -march=amdgcn -mcpu=gfx940 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX940 %s
; RUN: llc -O0 -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX1030 %s
; RUN: llc -O0 -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefixes=GFX1100 %s

; GFX906-LABEL: image_sample_test:
; GFX906-NOT: v_illegal
; GFX906: image_sample_lz

; GFX908-LABEL: image_sample_test:
; GFX908-NOT: v_illegal
; GFX908: image_sample_lz

; GFX90A-LABEL: image_sample_test:
; GFX90A-NOT: image_sample_lz
; GFX90A: v_illegal

; GFX940-LABEL: image_sample_test:
; GFX940-NOT: image_sample_lz
; GFX940: v_illegal

; GFX1030-LABEL: image_sample_test:
; GFX1030-NOT: v_illegal
; GFX1030: image_sample_lz

; GFX1100-LABEL: image_sample_test:
; GFX1100-NOT: v_illegal
; GFX1100: image_sample_lz

define amdgpu_kernel void @image_sample_test(<4 x float> addrspace(1)* %out, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4) {
  
  %result = tail call <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 15, float %arg1, float %arg2, <8 x i32> %arg3, <4 x i32> %arg4, i1 false, i32 0, i32 0)

  store <4 x float> %result, <4 x float> addrspace(1)* %out
  ret void
}

declare <4 x float> @llvm.amdgcn.image.sample.lz.2d.v4f32.f32(i32 immarg, float, float, <8 x i32>, <4 x i32>, i1 immarg, i32 immarg, i32 immarg)
