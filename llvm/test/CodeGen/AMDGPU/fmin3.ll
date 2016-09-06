; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare float @llvm.minnum.f32(float, float) nounwind readnone

; SI-LABEL: {{^}}test_fmin3_olt_0:
; SI: buffer_load_dword [[REGC:v[0-9]+]]
; SI: buffer_load_dword [[REGB:v[0-9]+]]
; SI: buffer_load_dword [[REGA:v[0-9]+]]
; SI: v_min3_f32 [[RESULT:v[0-9]+]], [[REGC]], [[REGB]], [[REGA]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @test_fmin3_olt_0(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) nounwind {
  %a = load volatile float, float addrspace(1)* %aptr, align 4
  %b = load volatile float, float addrspace(1)* %bptr, align 4
  %c = load volatile float, float addrspace(1)* %cptr, align 4
  %f0 = call float @llvm.minnum.f32(float %a, float %b) nounwind readnone
  %f1 = call float @llvm.minnum.f32(float %f0, float %c) nounwind readnone
  store float %f1, float addrspace(1)* %out, align 4
  ret void
}

; Commute operand of second fmin
; SI-LABEL: {{^}}test_fmin3_olt_1:
; SI: buffer_load_dword [[REGB:v[0-9]+]]
; SI: buffer_load_dword [[REGA:v[0-9]+]]
; SI: buffer_load_dword [[REGC:v[0-9]+]]
; SI: v_min3_f32 [[RESULT:v[0-9]+]], [[REGC]], [[REGB]], [[REGA]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
define void @test_fmin3_olt_1(float addrspace(1)* %out, float addrspace(1)* %aptr, float addrspace(1)* %bptr, float addrspace(1)* %cptr) nounwind {
  %a = load volatile float, float addrspace(1)* %aptr, align 4
  %b = load volatile float, float addrspace(1)* %bptr, align 4
  %c = load volatile float, float addrspace(1)* %cptr, align 4
  %f0 = call float @llvm.minnum.f32(float %a, float %b) nounwind readnone
  %f1 = call float @llvm.minnum.f32(float %c, float %f0) nounwind readnone
  store float %f1, float addrspace(1)* %out, align 4
  ret void
}
