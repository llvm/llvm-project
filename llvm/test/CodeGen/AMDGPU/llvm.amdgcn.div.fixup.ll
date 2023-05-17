; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

declare float @llvm.amdgcn.div.fixup.f32(float, float, float) nounwind readnone
declare double @llvm.amdgcn.div.fixup.f64(double, double, double) nounwind readnone

; GCN-LABEL: {{^}}test_div_fixup_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x13
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x1c
; SI-DAG: s_load_dword [[SC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x25

; VI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x4c
; VI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x70
; VI-DAG: s_load_dword [[SC:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0x94

; GCN-DAG: v_mov_b32_e32 [[VC:v[0-9]+]], [[SC]]
; GCN-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; GCN: v_div_fixup_f32 [[RESULT:v[0-9]+]], [[SA]], [[VB]], [[VC]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
define amdgpu_kernel void @test_div_fixup_f32(ptr addrspace(1) %out, [8 x i32], float %a, [8 x i32], float %b, [8 x i32], float %c) nounwind {
  %result = call float @llvm.amdgcn.div.fixup.f32(float %a, float %b, float %c) nounwind readnone
  store float %result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_div_fixup_f64:
; GCN: v_div_fixup_f64
define amdgpu_kernel void @test_div_fixup_f64(ptr addrspace(1) %out, double %a, double %b, double %c) nounwind {
  %result = call double @llvm.amdgcn.div.fixup.f64(double %a, double %b, double %c) nounwind readnone
  store double %result, ptr addrspace(1) %out, align 8
  ret void
}
