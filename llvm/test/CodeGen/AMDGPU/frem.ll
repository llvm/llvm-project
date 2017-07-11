; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs  < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=bonaire -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN -check-prefix=FUNC %s
; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=CI -check-prefix=GCN -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}frem_f32:
; GCN-DAG: buffer_load_dword [[X:v[0-9]+]], {{.*$}}
; GCN-DAG: buffer_load_dword [[Y:v[0-9]+]], {{.*}} offset:16
; GCN: v_div_scale_f32

; GCN: v_rcp_f32_e32
; GCN: v_fma_f32
; GCN: v_mul_f32_e32
; GCN: v_div_fmas_f32
; GCN: v_div_fixup_f32
; GCN: v_trunc_f32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_mad_f32 v{{[0-9]+}}, -v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: s_endpgm
define amdgpu_kernel void @frem_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                      float addrspace(1)* %in2) #0 {
   %gep2 = getelementptr float, float addrspace(1)* %in2, i32 4
   %r0 = load float, float addrspace(1)* %in1, align 4
   %r1 = load float, float addrspace(1)* %gep2, align 4
   %r2 = frem float %r0, %r1
   store float %r2, float addrspace(1)* %out, align 4
   ret void
}

; FUNC-LABEL: {{^}}unsafe_frem_f32:
; GCN: buffer_load_dword [[Y:v[0-9]+]], {{.*}} offset:16
; GCN: buffer_load_dword [[X:v[0-9]+]], {{.*}}
; GCN: v_rcp_f32_e32 [[INVY:v[0-9]+]], [[Y]]
; GCN: v_mul_f32_e32 [[DIV:v[0-9]+]], [[X]], [[INVY]]
; GCN: v_trunc_f32_e32 [[TRUNC:v[0-9]+]], [[DIV]]
; GCN: v_mad_f32 [[RESULT:v[0-9]+]], -[[TRUNC]], [[Y]], [[X]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @unsafe_frem_f32(float addrspace(1)* %out, float addrspace(1)* %in1,
                             float addrspace(1)* %in2) #1 {
   %gep2 = getelementptr float, float addrspace(1)* %in2, i32 4
   %r0 = load float, float addrspace(1)* %in1, align 4
   %r1 = load float, float addrspace(1)* %gep2, align 4
   %r2 = frem float %r0, %r1
   store float %r2, float addrspace(1)* %out, align 4
   ret void
}

; FUNC-LABEL: {{^}}frem_f64:
; GCN: buffer_load_dwordx2 [[Y:v\[[0-9]+:[0-9]+\]]], {{.*}}, 0
; GCN: buffer_load_dwordx2 [[X:v\[[0-9]+:[0-9]+\]]], {{.*}}, 0
; GCN-DAG: v_div_fmas_f64
; GCN-DAG: v_div_scale_f64
; GCN-DAG: v_mul_f64
; CI: v_trunc_f64_e32
; CI: v_mul_f64
; GCN: v_add_f64
; GCN: buffer_store_dwordx2
; GCN: s_endpgm
define amdgpu_kernel void @frem_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                      double addrspace(1)* %in2) #0 {
   %r0 = load double, double addrspace(1)* %in1, align 8
   %r1 = load double, double addrspace(1)* %in2, align 8
   %r2 = frem double %r0, %r1
   store double %r2, double addrspace(1)* %out, align 8
   ret void
}

; FUNC-LABEL: {{^}}unsafe_frem_f64:
; GCN: v_rcp_f64_e32
; GCN: v_mul_f64
; SI: v_bfe_u32
; CI: v_trunc_f64_e32
; GCN: v_fma_f64
; GCN: s_endpgm
define amdgpu_kernel void @unsafe_frem_f64(double addrspace(1)* %out, double addrspace(1)* %in1,
                             double addrspace(1)* %in2) #1 {
   %r0 = load double, double addrspace(1)* %in1, align 8
   %r1 = load double, double addrspace(1)* %in2, align 8
   %r2 = frem double %r0, %r1
   store double %r2, double addrspace(1)* %out, align 8
   ret void
}

define amdgpu_kernel void @frem_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in1,
                        <2 x float> addrspace(1)* %in2) #0 {
   %gep2 = getelementptr <2 x float>, <2 x float> addrspace(1)* %in2, i32 4
   %r0 = load <2 x float>, <2 x float> addrspace(1)* %in1, align 8
   %r1 = load <2 x float>, <2 x float> addrspace(1)* %gep2, align 8
   %r2 = frem <2 x float> %r0, %r1
   store <2 x float> %r2, <2 x float> addrspace(1)* %out, align 8
   ret void
}

define amdgpu_kernel void @frem_v4f32(<4 x float> addrspace(1)* %out, <4 x float> addrspace(1)* %in1,
                        <4 x float> addrspace(1)* %in2) #0 {
   %gep2 = getelementptr <4 x float>, <4 x float> addrspace(1)* %in2, i32 4
   %r0 = load <4 x float>, <4 x float> addrspace(1)* %in1, align 16
   %r1 = load <4 x float>, <4 x float> addrspace(1)* %gep2, align 16
   %r2 = frem <4 x float> %r0, %r1
   store <4 x float> %r2, <4 x float> addrspace(1)* %out, align 16
   ret void
}

define amdgpu_kernel void @frem_v2f64(<2 x double> addrspace(1)* %out, <2 x double> addrspace(1)* %in1,
                        <2 x double> addrspace(1)* %in2) #0 {
   %gep2 = getelementptr <2 x double>, <2 x double> addrspace(1)* %in2, i32 4
   %r0 = load <2 x double>, <2 x double> addrspace(1)* %in1, align 16
   %r1 = load <2 x double>, <2 x double> addrspace(1)* %gep2, align 16
   %r2 = frem <2 x double> %r0, %r1
   store <2 x double> %r2, <2 x double> addrspace(1)* %out, align 16
   ret void
}

attributes #0 = { nounwind "unsafe-fp-math"="false" }
attributes #1 = { nounwind "unsafe-fp-math"="true" }
