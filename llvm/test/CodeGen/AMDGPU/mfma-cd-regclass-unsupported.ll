; RUN: split-file %s %t
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 %t/explicit-auto.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VALUE
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 %t/explicit-auto.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VALUE
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 %t/bad-result.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 %t/bad-result.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 %t/bad-param.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 %t/bad-param.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 %t/bad-function.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 %t/bad-function.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 %t/bad-call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 %t/bad-call.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=PLACEMENT
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=1 %t/bad-value.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VALUE
; RUN: not llc -mtriple=amdgpu9.50-amd-amdhsa -global-isel=0 %t/bad-value.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VALUE
; RUN: not llc -mtriple=amdgpu9.08-amd-amdhsa -global-isel=0 %t/vgpr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VGPR
; RUN: not llc -mtriple=amdgpu9.08-amd-amdhsa -global-isel=1 -global-isel-abort=1 %t/vgpr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VGPR
; RUN: not --crash llc -mtriple=amdgpu9.0a-amd-amdhsa -global-isel=0 %t/scaled.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VARIANT
; RUN: not llc -mtriple=amdgpu9.0a-amd-amdhsa -global-isel=1 -global-isel-abort=1 %t/scaled.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=VARIANT
; RUN: not --crash llc -mtriple=amdgpu9.06-amd-amdhsa -global-isel=0 %t/agpr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TARGET
; RUN: not llc -mtriple=amdgpu9.06-amd-amdhsa -global-isel=1 -global-isel-abort=1 %t/agpr.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=TARGET
;
; VGPR: error: {{.*}}amdgpu-mfma-cd-regclass: VGPR C/D form requires gfx90a or later CDNA
; VARIANT: error: {{.*}}amdgpu-mfma-cd-regclass: target does not support scaled MFMA
; TARGET: error: {{.*}}amdgpu-mfma-cd-regclass: target does not support MFMA

;--- vgpr.ll
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
define <4 x float> @unsupported_vgpr(float %a, float %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="vgpr"
  ret <4 x float> %d
}

;--- scaled.ll
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32, i32, i32, i32, i32, i32)
define <4 x float> @unsupported_variant(<8 x i32> %a, <8 x i32> %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  ret <4 x float> %d
}

;--- agpr.ll
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
define <4 x float> @unsupported_agpr(float %a, float %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="agpr"
  ret <4 x float> %d
}

; VALUE: error: {{.*}}amdgpu-mfma-cd-regclass: expected agpr or vgpr
; PLACEMENT: error: {{.*}}amdgpu-mfma-cd-regclass: only valid on MFMA call sites

;--- bad-value.ll
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
define <4 x float> @bad_value(float %a, float %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="bad"
  ret <4 x float> %d
}

;--- bad-call.ll
declare void @external()
define void @bad_call() {
  call void @external() "amdgpu-mfma-cd-regclass"="agpr"
  ret void
}

;--- bad-function.ll
define void @bad_function() "amdgpu-mfma-cd-regclass"="agpr" {
  ret void
}

;--- bad-param.ll
declare void @external(i32)
define void @bad_param(i32 %x) {
  call void @external(i32 "amdgpu-mfma-cd-regclass"="agpr" %x)
  ret void
}

;--- bad-result.ll
declare i32 @external()
define i32 @bad_result() {
  %x = call "amdgpu-mfma-cd-regclass"="agpr" i32 @external()
  ret i32 %x
}

;--- explicit-auto.ll
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
define <4 x float> @explicit_auto(float %a, float %b, <4 x float> %c) {
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0) "amdgpu-mfma-cd-regclass"="auto"
  ret <4 x float> %d
}
