; REQUIRES: amdgpu-registered-target
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck %s

;;; f32 arithmetic

; CHECK-LABEL: {{^}}fadd_f32:
; CHECK: v_add_f32
define amdgpu_kernel void @fadd_f32(ptr addrspace(1) %out, float %a, float %b) {
  %r = call float @llvm.fadd.f32(float %a, float %b)
  store float %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}fsub_f32:
; CHECK: v_sub_f32
define amdgpu_kernel void @fsub_f32(ptr addrspace(1) %out, float %a, float %b) {
  %r = call float @llvm.fsub.f32(float %a, float %b)
  store float %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}fmul_f32:
; CHECK: v_mul_f32
define amdgpu_kernel void @fmul_f32(ptr addrspace(1) %out, float %a, float %b) {
  %r = call float @llvm.fmul.f32(float %a, float %b)
  store float %r, ptr addrspace(1) %out
  ret void
}

;;; f64 arithmetic

; CHECK-LABEL: {{^}}fadd_f64:
; CHECK: v_add_f64
define amdgpu_kernel void @fadd_f64(ptr addrspace(1) %out, double %a, double %b) {
  %r = call double @llvm.fadd.f64(double %a, double %b)
  store double %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}fmul_f64:
; CHECK: v_mul_f64
define amdgpu_kernel void @fmul_f64(ptr addrspace(1) %out, double %a, double %b) {
  %r = call double @llvm.fmul.f64(double %a, double %b)
  store double %r, ptr addrspace(1) %out
  ret void
}

;;; Conversions

; CHECK-LABEL: {{^}}fptrunc_f64_f32:
; CHECK: v_cvt_f32_f64
define amdgpu_kernel void @fptrunc_f64_f32(ptr addrspace(1) %out, double %a) {
  %r = call float @llvm.fptrunc.f32.f64(double %a)
  store float %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}fpext_f32_f64:
; CHECK: v_cvt_f64_f32
define amdgpu_kernel void @fpext_f32_f64(ptr addrspace(1) %out, float %a) {
  %r = call double @llvm.fpext.f64.f32(float %a)
  store double %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}sitofp_i32_f32:
; CHECK: v_cvt_f32_i32
define amdgpu_kernel void @sitofp_i32_f32(ptr addrspace(1) %out, i32 %a) {
  %r = call float @llvm.sitofp.f32.i32(i32 %a)
  store float %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}uitofp_i32_f32:
; CHECK: v_cvt_f32_u32
define amdgpu_kernel void @uitofp_i32_f32(ptr addrspace(1) %out, i32 %a) {
  %r = call float @llvm.uitofp.f32.i32(i32 %a)
  store float %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}fptosi_f32_i32:
; CHECK: v_cvt_i32_f32
define amdgpu_kernel void @fptosi_f32_i32(ptr addrspace(1) %out, float %a) {
  %r = call i32 @llvm.fptosi.i32.f32(float %a)
  store i32 %r, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}fptoui_f32_i32:
; CHECK: v_cvt_u32_f32
define amdgpu_kernel void @fptoui_f32_i32(ptr addrspace(1) %out, float %a) {
  %r = call i32 @llvm.fptoui.i32.f32(float %a)
  store i32 %r, ptr addrspace(1) %out
  ret void
}

;;; Compare

; CHECK-LABEL: {{^}}fcmp_oeq:
; CHECK: v_cmp_eq_f32
define amdgpu_kernel void @fcmp_oeq(ptr addrspace(1) %out, float %a, float %b) {
  %r = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
  %ext = zext i1 %r to i32
  store i32 %ext, ptr addrspace(1) %out
  ret void
}

;;; Fast-math flags

; fast on fadd.f32 -- same v_add_f32 on AMDGPU
; CHECK-LABEL: {{^}}fadd_fast_f32:
; CHECK: v_add_f32
define amdgpu_kernel void @fadd_fast_f32(ptr addrspace(1) %out, float %a, float %b) {
  %r = call fast float @llvm.fadd.f32(float %a, float %b)
  store float %r, ptr addrspace(1) %out
  ret void
}

; nnan nsz on fmul.f32 -- same v_mul_f32 on AMDGPU
; CHECK-LABEL: {{^}}fmul_nnan_nsz_f32:
; CHECK: v_mul_f32
define amdgpu_kernel void @fmul_nnan_nsz_f32(ptr addrspace(1) %out, float %a, float %b) {
  %r = call nnan nsz float @llvm.fmul.f32(float %a, float %b)
  store float %r, ptr addrspace(1) %out
  ret void
}

; contract on fmul+fadd -> v_fma_f32 or v_mac_f32 (FMA contraction)
; CHECK-LABEL: {{^}}fmadd_contract_f32:
; CHECK: v_{{fma|mac}}_f32
define amdgpu_kernel void @fmadd_contract_f32(ptr addrspace(1) %out, float %a, float %b, float %c) {
  %mul = call contract float @llvm.fmul.f32(float %a, float %b)
  %add = call contract float @llvm.fadd.f32(float %mul, float %c)
  store float %add, ptr addrspace(1) %out
  ret void
}

declare float @llvm.fadd.f32(float, float)
declare float @llvm.fsub.f32(float, float)
declare float @llvm.fmul.f32(float, float)
declare double @llvm.fadd.f64(double, double)
declare double @llvm.fmul.f64(double, double)
declare float @llvm.fptrunc.f32.f64(double)
declare double @llvm.fpext.f64.f32(float)
declare float @llvm.sitofp.f32.i32(i32)
declare float @llvm.uitofp.f32.i32(i32)
declare i32 @llvm.fptosi.i32.f32(float)
declare i32 @llvm.fptoui.i32.f32(float)
declare i1 @llvm.fcmp.f32(float, float, metadata)
