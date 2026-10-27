; RUN: split-file %s %t
; RUN: not llc -mtriple=amdgpu9.08-amd-amdhsa -global-isel=0 %t/vgpr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VGPR
; RUN: not llc -mtriple=amdgpu9.08-amd-amdhsa -global-isel=1 -global-isel-abort=1 %t/vgpr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VGPR
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -global-isel=0 %t/scaled.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VARIANT
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -global-isel=1 -global-isel-abort=1 %t/scaled.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VARIANT
; RUN: not llc -mtriple=amdgpu9.06-amd-amdhsa -global-isel=0 %t/agpr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TARGET
; RUN: not llc -mtriple=amdgpu9.06-amd-amdhsa -global-isel=1 -global-isel-abort=1 %t/agpr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TARGET
;
; VGPR: error: {{.*}}MFMA cd_regclass request cannot be satisfied: VGPR C/D form requires gfx90a or later CDNA
; VARIANT: error: {{.*}}MFMA cd_regclass request cannot be satisfied: instruction variant is unavailable on target
; TARGET: error: {{.*}}MFMA cd_regclass request cannot be satisfied: target does not support MFMA

;--- vgpr.ll
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
define <4 x float> @unsupported_vgpr(float %a, float %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) cd_regclass(vgpr)
  ret <4 x float> %d
}

;--- scaled.ll
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32, i32, i32, i32, i32, i32)
define <4 x float> @unsupported_variant(<8 x i32> %a, <8 x i32> %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0) cd_regclass(agpr)
  ret <4 x float> %d
}

;--- agpr.ll
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
define <4 x float> @unsupported_agpr(float %a, float %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) cd_regclass(agpr)
  ret <4 x float> %d
}
