; RUN: llc -mtriple=amdgcn -mcpu=fiji -O0 -stop-after=irtranslator -global-isel %s -o - | FileCheck %s

; Check flags are preserved for a regular instruction.
; CHECK-LABEL: name: fadd_nnan
; CHECK: nnan G_FADD
define amdgpu_kernel void @fadd_nnan(float %arg0, float %arg1) {
  %res = fadd nnan float %arg0, %arg1
  store float %res, ptr addrspace(1) undef
  ret void
}

; Check flags are preserved for a specially handled intrinsic
; CHECK-LABEL: name: fma_fast
; CHECK: nnan ninf nsz arcp contract afn reassoc G_FMA
define amdgpu_kernel void @fma_fast(float %arg0, float %arg1, float %arg2) {
  %res = call fast float @llvm.fma.f32(float %arg0, float %arg1, float %arg2)
  store float %res, ptr addrspace(1) undef
  ret void
}

; Check flags are preserved for an arbitrarry target intrinsic
; CHECK-LABEL: name: rcp_nsz
; CHECK: = nsz G_INTRINSIC intrinsic(@llvm.amdgcn.rcp), %{{[0-9]+}}(f32)
define amdgpu_kernel void @rcp_nsz(float %arg0) {
  %res = call nsz float @llvm.amdgcn.rcp.f32 (float %arg0)
  store float %res, ptr addrspace(1) undef
  ret void
}

declare float @llvm.fma.f32(float, float, float)
declare float @llvm.amdgcn.rcp.f32(float)
