; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG %s

declare float @llvm.r600.recipsqrt.ieee.f32(float) nounwind readnone

; EG-LABEL: {{^}}recipsqrt.ieee_f32:
; EG: RECIPSQRT_IEEE
define amdgpu_kernel void @recipsqrt.ieee_f32(ptr addrspace(1) %out, float %src) nounwind {
  %recipsqrt.ieee = call float @llvm.r600.recipsqrt.ieee.f32(float %src) nounwind readnone
  store float %recipsqrt.ieee, ptr addrspace(1) %out, align 4
  ret void
}

; TODO: Really these should be constant folded
; EG-LABEL: {{^}}recipsqrt.ieee_f32_constant_4.0
; EG: RECIPSQRT_IEEE
define amdgpu_kernel void @recipsqrt.ieee_f32_constant_4.0(ptr addrspace(1) %out) nounwind {
  %recipsqrt.ieee = call float @llvm.r600.recipsqrt.ieee.f32(float 4.0) nounwind readnone
  store float %recipsqrt.ieee, ptr addrspace(1) %out, align 4
  ret void
}

; EG-LABEL: {{^}}recipsqrt.ieee_f32_constant_100.0
; EG: RECIPSQRT_IEEE
define amdgpu_kernel void @recipsqrt.ieee_f32_constant_100.0(ptr addrspace(1) %out) nounwind {
  %recipsqrt.ieee = call float @llvm.r600.recipsqrt.ieee.f32(float 100.0) nounwind readnone
  store float %recipsqrt.ieee, ptr addrspace(1) %out, align 4
  ret void
}
