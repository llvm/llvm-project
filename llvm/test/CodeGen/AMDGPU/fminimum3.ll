; RUN: llc -mtriple=amdgcn -mcpu=gfx1200 -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN %s

; GCN-LABEL: {{^}}test_fminimum3_olt_0_f32:
; GCN: buffer_load_b32 [[REGC:v[0-9]+]]
; GCN: buffer_load_b32 [[REGB:v[0-9]+]]
; GCN: buffer_load_b32 [[REGA:v[0-9]+]]
; GCN: v_minimum3_f32 [[RESULT:v[0-9]+]], [[REGC]], [[REGB]], [[REGA]]
; GCN: buffer_store_b32 [[RESULT]],
define amdgpu_kernel void @test_fminimum3_olt_0_f32(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) {
  %a = load volatile float, ptr addrspace(1) %aptr, align 4
  %b = load volatile float, ptr addrspace(1) %bptr, align 4
  %c = load volatile float, ptr addrspace(1) %cptr, align 4
  %f0 = call float @llvm.minimum.f32(float %a, float %b)
  %f1 = call float @llvm.minimum.f32(float %f0, float %c)
  store float %f1, ptr addrspace(1) %out, align 4
  ret void
}

; Commute operand of second fminimum
; GCN-LABEL: {{^}}test_fminimum3_olt_1_f32:
; GCN: buffer_load_b32 [[REGB:v[0-9]+]]
; GCN: buffer_load_b32 [[REGA:v[0-9]+]]
; GCN: buffer_load_b32 [[REGC:v[0-9]+]]
; GCN: v_minimum3_f32 [[RESULT:v[0-9]+]], [[REGC]], [[REGB]], [[REGA]]
; GCN: buffer_store_b32 [[RESULT]],
define amdgpu_kernel void @test_fminimum3_olt_1_f32(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) {
  %a = load volatile float, ptr addrspace(1) %aptr, align 4
  %b = load volatile float, ptr addrspace(1) %bptr, align 4
  %c = load volatile float, ptr addrspace(1) %cptr, align 4
  %f0 = call float @llvm.minimum.f32(float %a, float %b)
  %f1 = call float @llvm.minimum.f32(float %c, float %f0)
  store float %f1, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_fminimum3_olt_0_f16:
; GCN: buffer_load_u16 [[REGC:v[0-9]+]]
; GCN: buffer_load_u16 [[REGB:v[0-9]+]]
; GCN: buffer_load_u16 [[REGA:v[0-9]+]]
; GCN: v_minimum3_f16 [[RESULT:v[0-9]+]], [[REGC]], [[REGB]], [[REGA]]
; GCN: buffer_store_b16 [[RESULT]],
define amdgpu_kernel void @test_fminimum3_olt_0_f16(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) {
  %a = load volatile half, ptr addrspace(1) %aptr, align 2
  %b = load volatile half, ptr addrspace(1) %bptr, align 2
  %c = load volatile half, ptr addrspace(1) %cptr, align 2
  %f0 = call half @llvm.minimum.f16(half %a, half %b)
  %f1 = call half @llvm.minimum.f16(half %f0, half %c)
  store half %f1, ptr addrspace(1) %out, align 2
  ret void
}

; GCN-LABEL: {{^}}test_fminimum3_olt_1_f16:
; GCN: buffer_load_u16 [[REGA:v[0-9]+]]
; GCN: buffer_load_u16 [[REGB:v[0-9]+]]
; GCN: buffer_load_u16 [[REGC:v[0-9]+]]
; GCN: v_minimum3_f16 [[RESULT:v[0-9]+]], [[REGC]], [[REGA]], [[REGB]]
; GCN: buffer_store_b16 [[RESULT]],
define amdgpu_kernel void @test_fminimum3_olt_1_f16(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) {
  %a = load volatile half, ptr addrspace(1) %aptr, align 2
  %b = load volatile half, ptr addrspace(1) %bptr, align 2
  %c = load volatile half, ptr addrspace(1) %cptr, align 2
  %f0 = call half @llvm.minimum.f16(half %a, half %b)
  %f1 = call half @llvm.minimum.f16(half %c, half %f0)
  store half %f1, ptr addrspace(1) %out, align 2
  ret void
}

; Checks whether the test passes; performMinMaxCombine() should not optimize vector patterns of minimum3
; since there are no pack instructions for fminimum3.
; GCN-LABEL: {{^}}no_fminimum3_v2f16:
; GCN: v_pk_minimum_f16 v0, v0, v1
; GCN: v_pk_minimum_f16 v0, v2, v0
; GCN: v_pk_minimum_f16 v0, v0, v3
; GCN-NEXT: s_setpc_b64
define <2 x half> @no_fminimum3_v2f16(<2 x half> %a, <2 x half> %b, <2 x half> %c, <2 x half> %d) {
entry:
  %min = call <2 x half> @llvm.minimum.v2f16(<2 x half> %a, <2 x half> %b)
  %min1 = call <2 x half> @llvm.minimum.v2f16(<2 x half> %c, <2 x half> %min)
  %res = call <2 x half> @llvm.minimum.v2f16(<2 x half> %min1, <2 x half> %d)
  ret <2 x half> %res
}

; GCN-LABEL: {{^}}no_fminimum3_olt_0_f64:
; GCN-COUNT-2: v_minimum_f64
define amdgpu_kernel void @no_fminimum3_olt_0_f64(ptr addrspace(1) %out, ptr addrspace(1) %aptr, ptr addrspace(1) %bptr, ptr addrspace(1) %cptr) {
  %a = load volatile double, ptr addrspace(1) %aptr, align 4
  %b = load volatile double, ptr addrspace(1) %bptr, align 4
  %c = load volatile double, ptr addrspace(1) %cptr, align 4
  %f0 = call double @llvm.minimum.f64(double %a, double %b)
  %f1 = call double @llvm.minimum.f64(double %f0, double %c)
  store double %f1, ptr addrspace(1) %out, align 4
  ret void
}

declare double @llvm.minimum.f64(double, double)
declare float @llvm.minimum.f32(float, float)
declare half @llvm.minimum.f16(half, half)
declare <2 x half> @llvm.minimum.v2f16(<2 x half>, <2 x half>)
