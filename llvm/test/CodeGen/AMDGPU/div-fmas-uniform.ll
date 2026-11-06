; RUN: llc -mtriple=amdgpu11.50 < %s | FileCheck %s --check-prefixes=CHECK,SALU
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgpu11.50 < %s | FileCheck %s --check-prefixes=CHECK,SALU
; RUN: llc -mtriple=amdgpu12.00 < %s | FileCheck %s --check-prefixes=CHECK,SALU
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgpu12.00 < %s | FileCheck %s --check-prefixes=CHECK,SALU
; RUN: llc -mtriple=amdgpu12.50 < %s | FileCheck %s --check-prefixes=CHECK,SALU
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgpu12.50 < %s | FileCheck %s --check-prefixes=CHECK,SALU
; RUN: llc -mtriple=amdgpu11.00 < %s | FileCheck %s --check-prefixes=CHECK,VALU
; RUN: llc -global-isel -global-isel-abort=1 -mtriple=amdgpu11.00 < %s | FileCheck %s --check-prefixes=CHECK,VALU

; Kernel arguments are uniform and can use scalar FMA where supported.

define amdgpu_kernel void @uniform_ieee(ptr addrspace(1) %out, float %a, float %b, float %c) #0 {
; CHECK-LABEL: uniform_ieee:
; SALU-NOT: v_fma
; SALU: s_fmac_f32
; SALU-NOT: v_fma
; VALU: v_fma{{c?}}_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  store float %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_flush_outputs(ptr addrspace(1) %out, float %a, float %b, float %c) #1 {
; CHECK-LABEL: uniform_flush_outputs:
; SALU-NOT: v_fma
; SALU: s_fmac_f32
; SALU-NOT: v_fma
; VALU: v_fma{{c?}}_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  store float %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_override(ptr addrspace(1) %out, float %a, float %b, float %c) #2 {
; CHECK-LABEL: uniform_override:
; SALU-NOT: v_fma
; SALU: s_fmac_f32
; SALU-NOT: v_fma
; VALU: v_fma{{c?}}_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  store float %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_true(ptr addrspace(1) %out, float %a, float %b, float %c) #0 {
; CHECK-LABEL: uniform_true:
; CHECK: v_div_fmas_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 true)
  store float %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_runtime(ptr addrspace(1) %out, float %a, float %b, float %c, i1 %scale) #0 {
; CHECK-LABEL: uniform_runtime:
; CHECK: v_div_fmas_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 %scale)
  store float %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_flush_inputs(ptr addrspace(1) %out, float %a, float %b, float %c) #3 {
; CHECK-LABEL: uniform_flush_inputs:
; CHECK: v_div_fmas_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  store float %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_dynamic_inputs(ptr addrspace(1) %out, float %a, float %b, float %c) #4 {
; CHECK-LABEL: uniform_dynamic_inputs:
; CHECK: v_div_fmas_f32
  %r = call float @llvm.amdgcn.div.fmas.f32(float %a, float %b, float %c, i1 false)
  store float %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @divergent_operand(ptr addrspace(1) %out, float %b, float %c) #0 {
; CHECK-LABEL: divergent_operand:
; CHECK: v_fma{{c?}}_f32
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %x = uitofp i32 %tid to float
  %r = call float @llvm.amdgcn.div.fmas.f32(float %x, float %b, float %c, i1 false)
  store float %r, ptr addrspace(1) %out
  ret void
}

define amdgpu_kernel void @uniform_f64(ptr addrspace(1) %out, double %a, double %b, double %c) #0 {
; CHECK-LABEL: uniform_f64:
; CHECK: v_fma{{c?}}_f64
  %r = call double @llvm.amdgcn.div.fmas.f64(double %a, double %b, double %c, i1 false)
  store double %r, ptr addrspace(1) %out
  ret void
}

attributes #0 = { denormal_fpenv(ieee) }
attributes #1 = { denormal_fpenv(preservesign|ieee) }
attributes #2 = { denormal_fpenv(preservesign, float: ieee) }
attributes #3 = { denormal_fpenv(ieee|preservesign) }
attributes #4 = { denormal_fpenv(ieee|dynamic) }
