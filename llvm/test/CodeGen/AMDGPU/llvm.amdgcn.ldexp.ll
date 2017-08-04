; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare float @llvm.amdgcn.ldexp.f32(float, i32) nounwind readnone
declare double @llvm.amdgcn.ldexp.f64(double, i32) nounwind readnone

; SI-LABEL: {{^}}test_ldexp_f32:
; SI: v_ldexp_f32
; SI: s_endpgm
define amdgpu_kernel void @test_ldexp_f32(float addrspace(1)* %out, float %a, i32 %b) nounwind {
  %result = call float @llvm.amdgcn.ldexp.f32(float %a, i32 %b) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_ldexp_f64:
; SI: v_ldexp_f64
; SI: s_endpgm
define amdgpu_kernel void @test_ldexp_f64(double addrspace(1)* %out, double %a, i32 %b) nounwind {
  %result = call double @llvm.amdgcn.ldexp.f64(double %a, i32 %b) nounwind readnone
  store double %result, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_ldexp_undef_f32:
; SI-NOT: v_ldexp_f32
define amdgpu_kernel void @test_ldexp_undef_f32(float addrspace(1)* %out, i32 %b) nounwind {
  %result = call float @llvm.amdgcn.ldexp.f32(float undef, i32 %b) nounwind readnone
  store float %result, float addrspace(1)* %out, align 4
  ret void
}
