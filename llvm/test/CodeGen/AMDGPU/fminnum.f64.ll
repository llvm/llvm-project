; RUN: llc -march=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare double @llvm.minnum.f64(double, double) #0
declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>) #0
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>) #0
declare <8 x double> @llvm.minnum.v8f64(<8 x double>, <8 x double>) #0
declare <16 x double> @llvm.minnum.v16f64(<16 x double>, <16 x double>) #0

; FUNC-LABEL: {{^}}test_fmin_f64_ieee:
; SI: s_load_dwordx2 [[A:s\[[0-9]+:[0-9]+\]]]
; SI: s_load_dwordx2 [[B:s\[[0-9]+:[0-9]+\]]]
; SI-DAG: v_max_f64 [[QUIETA:v\[[0-9]+:[0-9]+\]]], [[A]], [[A]]
; SI-DAG: v_max_f64 [[QUIETB:v\[[0-9]+:[0-9]+\]]], [[B]], [[B]]
; SI: v_min_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[QUIETA]], [[QUIETB]]
define amdgpu_kernel void @test_fmin_f64_ieee([8 x i32], double %a, [8 x i32], double %b) nounwind {
  %val = call double @llvm.minnum.f64(double %a, double %b) #0
  store double %val, double addrspace(1)* undef, align 8
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_f64_no_ieee:
; SI: ds_read_b64 [[VAL0:v\[[0-9]+:[0-9]+\]]]
; SI: ds_read_b64 [[VAL1:v\[[0-9]+:[0-9]+\]]]
; SI-NOT: [[VAL0]]
; SI-NOT: [[VAL1]]
; SI: v_min_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VAL0]], [[VAL1]]
; SI-NOT: [[RESULT]]
; SI: ds_write_b64 v{{[0-9]+}}, [[RESULT]]
define amdgpu_ps void @test_fmin_f64_no_ieee() nounwind {
  %a = load volatile double, double addrspace(3)* undef
  %b = load volatile double, double addrspace(3)* undef
  %val = call double @llvm.minnum.f64(double %a, double %b) #0
  store volatile double %val, double addrspace(3)* undef
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_v2f64:
; SI: v_min_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_v2f64(<2 x double> addrspace(1)* %out, <2 x double> %a, <2 x double> %b) nounwind {
  %val = call <2 x double> @llvm.minnum.v2f64(<2 x double> %a, <2 x double> %b) #0
  store <2 x double> %val, <2 x double> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_v4f64:
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_v4f64(<4 x double> addrspace(1)* %out, <4 x double> %a, <4 x double> %b) nounwind {
  %val = call <4 x double> @llvm.minnum.v4f64(<4 x double> %a, <4 x double> %b) #0
  store <4 x double> %val, <4 x double> addrspace(1)* %out, align 32
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_v8f64:
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_v8f64(<8 x double> addrspace(1)* %out, <8 x double> %a, <8 x double> %b) nounwind {
  %val = call <8 x double> @llvm.minnum.v8f64(<8 x double> %a, <8 x double> %b) #0
  store <8 x double> %val, <8 x double> addrspace(1)* %out, align 64
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_v16f64:
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
; SI: v_min_f64
define amdgpu_kernel void @test_fmin_v16f64(<16 x double> addrspace(1)* %out, <16 x double> %a, <16 x double> %b) nounwind {
  %val = call <16 x double> @llvm.minnum.v16f64(<16 x double> %a, <16 x double> %b) #0
  store <16 x double> %val, <16 x double> addrspace(1)* %out, align 128
  ret void
}

attributes #0 = { nounwind readnone }
