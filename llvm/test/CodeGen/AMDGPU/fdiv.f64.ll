; RUN: llc -march=amdgcn -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN %s


; GCN-LABEL: {{^}}fdiv_f64:
; GCN-DAG: buffer_load_dwordx2 [[NUM:v\[[0-9]+:[0-9]+\]]], off, {{s\[[0-9]+:[0-9]+\]}}, 0
; GCN-DAG: buffer_load_dwordx2 [[DEN:v\[[0-9]+:[0-9]+\]]], off, {{s\[[0-9]+:[0-9]+\]}}, 0 offset:8
; CI-DAG: v_div_scale_f64 [[SCALE0:v\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, [[DEN]], [[DEN]], [[NUM]]
; CI-DAG: v_div_scale_f64 [[SCALE1:v\[[0-9]+:[0-9]+\]]], vcc, [[NUM]], [[DEN]], [[NUM]]

; Check for div_scale bug workaround on SI
; SI-DAG: v_div_scale_f64 [[SCALE0:v\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, [[DEN]], [[DEN]], [[NUM]]
; SI-DAG: v_div_scale_f64 [[SCALE1:v\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, [[NUM]], [[DEN]], [[NUM]]

; GCN-DAG: v_rcp_f64_e32 [[RCP_SCALE0:v\[[0-9]+:[0-9]+\]]], [[SCALE0]]

; SI-DAG: v_cmp_eq_u32_e32 vcc, {{v[0-9]+}}, {{v[0-9]+}}
; SI-DAG: v_cmp_eq_u32_e64 [[CMP0:s\[[0-9]+:[0-9]+\]]], {{v[0-9]+}}, {{v[0-9]+}}
; SI-DAG: s_xor_b64 vcc, [[CMP0]], vcc

; GCN-DAG: v_fma_f64 [[FMA0:v\[[0-9]+:[0-9]+\]]], -[[SCALE0]], [[RCP_SCALE0]], 1.0
; GCN-DAG: v_fma_f64 [[FMA1:v\[[0-9]+:[0-9]+\]]], [[RCP_SCALE0]], [[FMA0]], [[RCP_SCALE0]]
; GCN-DAG: v_fma_f64 [[FMA2:v\[[0-9]+:[0-9]+\]]], -[[SCALE0]], [[FMA1]], 1.0
; GCN-DAG: v_fma_f64 [[FMA3:v\[[0-9]+:[0-9]+\]]], [[FMA1]], [[FMA2]], [[FMA1]]
; GCN-DAG: v_mul_f64 [[MUL:v\[[0-9]+:[0-9]+\]]], [[SCALE1]], [[FMA3]]
; GCN-DAG: v_fma_f64 [[FMA4:v\[[0-9]+:[0-9]+\]]], -[[SCALE0]], [[MUL]], [[SCALE1]]
; GCN: v_div_fmas_f64 [[FMAS:v\[[0-9]+:[0-9]+\]]], [[FMA4]], [[FMA3]], [[MUL]]
; GCN: v_div_fixup_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[FMAS]], [[DEN]], [[NUM]]
; GCN: buffer_store_dwordx2 [[RESULT]]
; GCN: s_endpgm
define amdgpu_kernel void @fdiv_f64(double addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %gep.1 = getelementptr double, double addrspace(1)* %in, i32 1
  %num = load volatile double, double addrspace(1)* %in
  %den = load volatile double, double addrspace(1)* %gep.1
  %result = fdiv double %num, %den
  store double %result, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fdiv_f64_afn:
; GCN: v_rcp_f64_e32 v[4:5], v[2:3]
; GCN: v_fma_f64 v[6:7], -v[2:3], v[4:5], 1.0
; GCN: v_fma_f64 v[4:5], v[6:7], v[4:5], v[4:5]
; GCN: v_fma_f64 v[6:7], -v[2:3], v[4:5], 1.0
; GCN: v_fma_f64 v[4:5], v[6:7], v[4:5], v[4:5]
; GCN: v_mul_f64 v[6:7], v[0:1], v[4:5]
; GCN: v_fma_f64 v[0:1], -v[2:3], v[6:7], v[0:1]
; GCN: v_fma_f64 v[0:1], v[0:1], v[4:5], v[6:7]
; GCN: s_setpc_b64
define double @v_fdiv_f64_afn(double %x, double %y) #0 {
  %result = fdiv afn double %x, %y
  ret double %result
}

; GCN-LABEL: {{^}}v_rcp_f64_afn:
; GCN: v_rcp_f64_e32 v[2:3], v[0:1]
; GCN: v_fma_f64 v[4:5], -v[0:1], v[2:3], 1.0
; GCN: v_fma_f64 v[2:3], v[4:5], v[2:3], v[2:3]
; GCN: v_fma_f64 v[4:5], -v[0:1], v[2:3], 1.0
; GCN: v_fma_f64 v[2:3], v[4:5], v[2:3], v[2:3]
; GCN: v_fma_f64 v[0:1], -v[0:1], v[2:3], 1.0
; GCN: v_fma_f64 v[0:1], v[0:1], v[2:3], v[2:3]
; GCN: s_setpc_b64
define double @v_rcp_f64_afn(double %x) #0 {
  %result = fdiv afn double 1.0, %x
  ret double %result
}

; GCN-LABEL: {{^}}fdiv_f64_s_v:
define amdgpu_kernel void @fdiv_f64_s_v(double addrspace(1)* %out, double addrspace(1)* %in, double %num) #0 {
  %den = load double, double addrspace(1)* %in
  %result = fdiv double %num, %den
  store double %result, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fdiv_f64_v_s:
define amdgpu_kernel void @fdiv_f64_v_s(double addrspace(1)* %out, double addrspace(1)* %in, double %den) #0 {
  %num = load double, double addrspace(1)* %in
  %result = fdiv double %num, %den
  store double %result, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}fdiv_f64_s_s:
define amdgpu_kernel void @fdiv_f64_s_s(double addrspace(1)* %out, double %num, double %den) #0 {
  %result = fdiv double %num, %den
  store double %result, double addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fdiv_v2f64:
define amdgpu_kernel void @v_fdiv_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %in) #0 {
  %gep.1 = getelementptr <2 x double>, <2 x double> addrspace(1)* %in, i32 1
  %num = load <2 x double>, <2 x double> addrspace(1)* %in
  %den = load <2 x double>, <2 x double> addrspace(1)* %gep.1
  %result = fdiv <2 x double> %num, %den
  store <2 x double> %result, <2 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_fdiv_v2f64:
define amdgpu_kernel void @s_fdiv_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %num, <2 x double> %den) {
  %result = fdiv <2 x double> %num, %den
  store <2 x double> %result, <2 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}v_fdiv_v4f64:
define amdgpu_kernel void @v_fdiv_v4f64(<4 x double> addrspace(1)* %out, <4 x double> addrspace(1)* %in) #0 {
  %gep.1 = getelementptr <4 x double>, <4 x double> addrspace(1)* %in, i32 1
  %num = load <4 x double>, <4 x double> addrspace(1)* %in
  %den = load <4 x double>, <4 x double> addrspace(1)* %gep.1
  %result = fdiv <4 x double> %num, %den
  store <4 x double> %result, <4 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}s_fdiv_v4f64:
define amdgpu_kernel void @s_fdiv_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %num, <4 x double> %den) #0 {
  %result = fdiv <4 x double> %num, %den
  store <4 x double> %result, <4 x double> addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}div_fast_2_x_pat_f64:
; GCN: v_mul_f64 [[MUL:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, 0.5
; GCN: buffer_store_dwordx2 [[MUL]]
define amdgpu_kernel void @div_fast_2_x_pat_f64(double addrspace(1)* %out) #1 {
  %x = load double, double addrspace(1)* undef
  %rcp = fdiv fast double %x, 2.0
  store double %rcp, double addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}div_fast_k_x_pat_f64:
; GCN-DAG: s_mov_b32 s[[K_LO:[0-9]+]], 0x9999999a
; GCN-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 0x3fb99999
; GCN: v_mul_f64 [[MUL:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}
; GCN: buffer_store_dwordx2 [[MUL]]
define amdgpu_kernel void @div_fast_k_x_pat_f64(double addrspace(1)* %out) #1 {
  %x = load double, double addrspace(1)* undef
  %rcp = fdiv fast double %x, 10.0
  store double %rcp, double addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}div_fast_neg_k_x_pat_f64:
; GCN-DAG: s_mov_b32 s[[K_LO:[0-9]+]], 0x9999999a
; GCN-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 0xbfb99999
; GCN: v_mul_f64 [[MUL:v\[[0-9]+:[0-9]+\]]], v{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}
; GCN: buffer_store_dwordx2 [[MUL]]
define amdgpu_kernel void @div_fast_neg_k_x_pat_f64(double addrspace(1)* %out) #1 {
  %x = load double, double addrspace(1)* undef
  %rcp = fdiv fast double %x, -10.0
  store double %rcp, double addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind "unsafe-fp-math"="true" }
