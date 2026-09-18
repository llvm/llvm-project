; RUN: not llc -global-isel=0 -mtriple=amdgpu9.50 -filetype=null < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -global-isel-abort=0 -mtriple=amdgpu9.50 -filetype=null < %s 2>&1 | FileCheck %s
;
; CHECK: llvm.amdgcn.wmma.f16.16x16x16.f16.tied requires target feature 'wmma-256b-insts'
; CHECK: llvm.amdgcn.wmma.f32.16x16x16.f16 requires target feature 'wmma-256b-insts|wmma-128b-insts'
; CHECK: llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8 requires target feature 'wmma-128b-insts'
; CHECK: llvm.amdgcn.wmma.f32.16x16x4.f32 requires target feature 'gfx1250-insts'
; CHECK: llvm.amdgcn.wmma.f64.16x16x4.f64 requires target feature 'gfx1251-gemm-insts'
; CHECK: llvm.amdgcn.wmma.f32.16x16x32.bf16 requires target feature 'wmma-n16-insts'

define <16 x half> @wmma_256b(<16 x half> %a, <16 x half> %b, <16 x half> %c) {
  %result = call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.tied(<16 x half> %a, <16 x half> %b, <16 x half> %c, i1 false)
  ret <16 x half> %result
}

define <8 x float> @wmma_256b_or_128b(<16 x half> %a, <16 x half> %b, <8 x float> %c) {
  %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16(<16 x half> %a, <16 x half> %b, <8 x float> %c)
  ret <8 x float> %result
}

define <8 x float> @wmma_128b(<2 x i32> %a, <2 x i32> %b, <8 x float> %c) {
  %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8(<2 x i32> %a, <2 x i32> %b, <8 x float> %c)
  ret <8 x float> %result
}

define <8 x float> @gfx1250(<2 x float> %a, <2 x float> %b, <8 x float> %c) {
  %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32(<2 x float> %a, <2 x float> %b, i16 0, <8 x float> %c, i1 false, i1 false)
  ret <8 x float> %result
}

define <8 x double> @gfx1251_gemm(<2 x double> %a, <2 x double> %b, <8 x double> %c) {
  %result = call <8 x double> @llvm.amdgcn.wmma.f64.16x16x4.f64.v8f64.v2f64(i1 false, <2 x double> %a, i1 false, <2 x double> %b, i16 0, <8 x double> %c, i1 false, i1 false)
  ret <8 x double> %result
}

define <8 x float> @wmma_n16(<16 x bfloat> %a, <16 x bfloat> %b, <8 x float> %c) {
  %result = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x32.bf16.v8f32.v16bf16(<16 x bfloat> %a, <16 x bfloat> %b, i16 0, <8 x float> %c, i1 false, i1 false)
  ret <8 x float> %result
}
