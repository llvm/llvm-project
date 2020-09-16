; RUN: not --crash llc -global-isel < %s -march=amdgcn -mcpu=gfx908 -verify-machineinstrs 2>&1 | FileCheck %s -check-prefix=GFX908

declare float @llvm.amdgcn.global.atomic.fadd.f32.p1f32.f32(float addrspace(1)* nocapture, float) #0

; GFX908: error: {{.*}} return versions of fp atomics not supported

define float @global_atomic_fadd_f32_rtn(float addrspace(1)* %ptr, float %data) {
  %ret = call float @llvm.amdgcn.global.atomic.fadd.f32.p1f32.f32(float addrspace(1)* %ptr, float %data)
  ret float %ret
}
