; RUN: llc -mtriple=amdgcn--amdpal -mcpu=fiji -print-after=amdgpu-isel -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=SI %s

; This checks that the -print-after of MIR containing a target custom pseudo
; value works correctly.

; SI: TargetCustom

target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-A5"
target triple = "amdgcn--amdpal"

define dllexport amdgpu_ps <2 x float> @_amdgpu_ps_main(i32 inreg, i32 inreg, i32 inreg, i32 inreg, <2 x float>, <2 x float>, <2 x float>, <3 x float>, <2 x float>, <2 x float>, <2 x float>, float, float, float, float, float, i32, i32, i32, i32) local_unnamed_addr {
.entry:
  %res = call <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32 3, float undef, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0)
  ret <2 x float> %res
}

declare <2 x float> @llvm.amdgcn.image.sample.1d.v2f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32)
