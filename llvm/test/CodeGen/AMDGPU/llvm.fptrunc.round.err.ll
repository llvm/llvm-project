; RUN: split-file %s %t

; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1030 -filetype=null %t/f16-f64-err.ll 2>&1 | FileCheck --ignore-case --check-prefix=F16-F64-FAIL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1030 -filetype=null %t/f16-f64-err.ll 2>&1 | FileCheck --ignore-case --check-prefix=F16-F64-FAIL %s

; TODO: check for GISEL when bfloat is supported.
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1030 -filetype=null %t/bf16-f32-err.ll 2>&1 | FileCheck --ignore-case --check-prefix=BF16-F32-FAIL %s
; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1030 -filetype=null %t/bf16-f64-err.ll 2>&1 | FileCheck --ignore-case --check-prefix=BF16-F64-FAIL %s

; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1030 -filetype=null %t/f16-f32-tonearestaway-err.ll 2>&1 | FileCheck --ignore-case --check-prefix=F16-F32-TONEARESTAWAY-FAIL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1030 -filetype=null %t/f16-f32-tonearestaway-err.ll 2>&1 | FileCheck --ignore-case --check-prefix=F16-F32-TONEARESTAWAY-FAIL %s

; RUN: not --crash llc -global-isel=0 -mtriple=amdgcn -mcpu=gfx1030 -filetype=null %t/f32-f64-tonearestaway-err.ll 2>&1 | FileCheck --ignore-case --check-prefix=F32-F64-TONEARESTAWAY-FAIL %s
; RUN: not --crash llc -global-isel=1 -mtriple=amdgcn -mcpu=gfx1030 -filetype=null %t/f32-f64-tonearestaway-err.ll 2>&1 | FileCheck --ignore-case --check-prefix=F32-F64-TONEARESTAWAY-FAIL %s

;--- f16-f64-err.ll
define amdgpu_gs void @test_fptrunc_round_f16_f64(double %a, ptr addrspace(1) %out) {
; F16-F64-FAIL: LLVM ERROR: Cannot select
  %res = call half @llvm.fptrunc.round.f16.f64(double %a, metadata !"round.upward")
  store half %res, ptr addrspace(1) %out, align 2
  ret void
}

;--- bf16-f32-err.ll
define amdgpu_gs void @test_fptrunc_round_bf16_f32(float %a, ptr addrspace(1) %out) {
; BF16-F32-FAIL: LLVM ERROR: Cannot select
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(float %a, metadata !"round.towardzero")
  store bfloat %res, ptr addrspace(1) %out, align 2
  ret void
}

;--- bf16-f64-err.ll
define amdgpu_gs void @test_fptrunc_round_bf16_f64(double %a, ptr addrspace(1) %out) {
; BF16-F64-FAIL: LLVM ERROR: Cannot select
  %res = call bfloat @llvm.fptrunc.round.bf16.f32(double %a, metadata !"round.tonearest")
  store bfloat %res, ptr addrspace(1) %out, align 2
  ret void
}

;--- f16-f32-tonearestaway-err.ll
define amdgpu_gs void @test_fptrunc_round_f16_f32_tonearestaway(float %a, ptr addrspace(1) %out) {
; F16-F32-TONEARESTAWAY-FAIL: LLVM ERROR: Cannot select
  %res = call half @llvm.fptrunc.round.f16.f32(float %a, metadata !"round.tonearestaway")
  store half %res, ptr addrspace(1) %out, align 2
  ret void
}

;--- f32-f64-tonearestaway-err.ll
define amdgpu_gs void @test_fptrunc_round_f32_f64_tonearestaway(double %a, ptr addrspace(1) %out) {
; F32-F64-TONEARESTAWAY-FAIL: LLVM ERROR: Cannot select
  %res = call float @llvm.fptrunc.round.f32.f64(double %a, metadata !"round.tonearestaway")
  store float %res, ptr addrspace(1) %out, align 4
  ret void
}
