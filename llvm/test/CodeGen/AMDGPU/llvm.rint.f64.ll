; RUN: llc -mtriple=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=FUNC %s
; RUN: llc -mtriple=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}rint_f64:
; CI: v_rndne_f64_e32

; SI-DAG: v_add_f64
; SI-DAG: v_add_f64
; SI-DAG: v_cmp_gt_f64_e64
; SI: v_cndmask_b32
; SI: v_cndmask_b32
; SI: s_endpgm
define amdgpu_kernel void @rint_f64(ptr addrspace(1) %out, double %in) {
entry:
  %0 = call double @llvm.rint.f64(double %in)
  store double %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}rint_v2f64:
; CI: v_rndne_f64_e32
; CI: v_rndne_f64_e32
define amdgpu_kernel void @rint_v2f64(ptr addrspace(1) %out, <2 x double> %in) {
entry:
  %0 = call <2 x double> @llvm.rint.v2f64(<2 x double> %in)
  store <2 x double> %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}rint_v4f64:
; CI: v_rndne_f64_e32
; CI: v_rndne_f64_e32
; CI: v_rndne_f64_e32
; CI: v_rndne_f64_e32
define amdgpu_kernel void @rint_v4f64(ptr addrspace(1) %out, <4 x double> %in) {
entry:
  %0 = call <4 x double> @llvm.rint.v4f64(<4 x double> %in)
  store <4 x double> %0, ptr addrspace(1) %out
  ret void
}


declare double @llvm.rint.f64(double) #0
declare <2 x double> @llvm.rint.v2f64(<2 x double>) #0
declare <4 x double> @llvm.rint.v4f64(<4 x double>) #0
