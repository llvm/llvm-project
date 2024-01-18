; RUN: llc -mtriple=amdgcn -mcpu=tahiti < %s | FileCheck -check-prefixes=GCN,GFX678 %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefixes=GCN,GFX678 %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,GFX9 %s

declare double @llvm.minnum.f64(double, double) #0
declare <2 x double> @llvm.minnum.v2f64(<2 x double>, <2 x double>) #0
declare <4 x double> @llvm.minnum.v4f64(<4 x double>, <4 x double>) #0
declare <8 x double> @llvm.minnum.v8f64(<8 x double>, <8 x double>) #0
declare <16 x double> @llvm.minnum.v16f64(<16 x double>, <16 x double>) #0

; GCN-LABEL: {{^}}test_fmin_f64_ieee_noflush:
; GCN: s_load_dwordx2 [[A:s\[[0-9]+:[0-9]+\]]]
; GCN: s_load_dwordx2 [[B:s\[[0-9]+:[0-9]+\]]]

; GCN-DAG: v_max_f64 [[QUIETA:v\[[0-9]+:[0-9]+\]]], [[A]], [[A]]
; GCN-DAG: v_max_f64 [[QUIETB:v\[[0-9]+:[0-9]+\]]], [[B]], [[B]]

; GCN: v_min_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[QUIETB]], [[QUIETA]]
define amdgpu_kernel void @test_fmin_f64_ieee_noflush([8 x i32], double %a, [8 x i32], double %b) #1 {
  %val = call double @llvm.minnum.f64(double %a, double %b) #0
  store double %val, ptr addrspace(1) undef, align 8
  ret void
}

; GCN-LABEL: {{^}}test_fmin_f64_ieee_flush:
; GCN: s_load_dwordx2 [[A:s\[[0-9]+:[0-9]+\]]]
; GCN: s_load_dwordx2 [[B:s\[[0-9]+:[0-9]+\]]]
; GFX678-DAG: v_mul_f64 [[QUIETA:v\[[0-9]+:[0-9]+\]]], 1.0, [[A]]
; GFX678-DAG: v_mul_f64 [[QUIETB:v\[[0-9]+:[0-9]+\]]], 1.0, [[B]]

; GFX9-DAG: v_max_f64 [[QUIETA:v\[[0-9]+:[0-9]+\]]], [[A]], [[A]]
; GFX9-DAG: v_max_f64 [[QUIETB:v\[[0-9]+:[0-9]+\]]], [[B]], [[B]]

; GCN: v_min_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[QUIETB]], [[QUIETA]]
define amdgpu_kernel void @test_fmin_f64_ieee_flush([8 x i32], double %a, [8 x i32], double %b) #2 {
  %val = call double @llvm.minnum.f64(double %a, double %b) #0
  store double %val, ptr addrspace(1) undef, align 8
  ret void
}

; GCN-LABEL: {{^}}test_fmin_f64_no_ieee:
; GCN: ds_read_b64 [[VAL0:v\[[0-9]+:[0-9]+\]]]
; GCN: ds_read_b64 [[VAL1:v\[[0-9]+:[0-9]+\]]]
; GCN-NOT: [[VAL0]]
; GCN-NOT: [[VAL1]]
; GCN: v_min_f64 [[RESULT:v\[[0-9]+:[0-9]+\]]], [[VAL0]], [[VAL1]]
; GCN-NOT: [[RESULT]]
; GCN: ds_write_b64 v{{[0-9]+}}, [[RESULT]]
define amdgpu_ps void @test_fmin_f64_no_ieee() nounwind {
  %a = load volatile double, ptr addrspace(3) undef
  %b = load volatile double, ptr addrspace(3) undef
  %val = call double @llvm.minnum.f64(double %a, double %b) #0
  store volatile double %val, ptr addrspace(3) undef
  ret void
}

; GCN-LABEL: {{^}}test_fmin_v2f64:
; GCN: v_min_f64
; GCN: v_min_f64
define amdgpu_kernel void @test_fmin_v2f64(ptr addrspace(1) %out, <2 x double> %a, <2 x double> %b) nounwind {
  %val = call <2 x double> @llvm.minnum.v2f64(<2 x double> %a, <2 x double> %b) #0
  store <2 x double> %val, ptr addrspace(1) %out, align 16
  ret void
}

; GCN-LABEL: {{^}}test_fmin_v4f64:
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
define amdgpu_kernel void @test_fmin_v4f64(ptr addrspace(1) %out, <4 x double> %a, <4 x double> %b) nounwind {
  %val = call <4 x double> @llvm.minnum.v4f64(<4 x double> %a, <4 x double> %b) #0
  store <4 x double> %val, ptr addrspace(1) %out, align 32
  ret void
}

; GCN-LABEL: {{^}}test_fmin_v8f64:
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
define amdgpu_kernel void @test_fmin_v8f64(ptr addrspace(1) %out, <8 x double> %a, <8 x double> %b) nounwind {
  %val = call <8 x double> @llvm.minnum.v8f64(<8 x double> %a, <8 x double> %b) #0
  store <8 x double> %val, ptr addrspace(1) %out, align 64
  ret void
}

; GCN-LABEL: {{^}}test_fmin_v16f64:
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
; GCN: v_min_f64
define amdgpu_kernel void @test_fmin_v16f64(ptr addrspace(1) %out, <16 x double> %a, <16 x double> %b) nounwind {
  %val = call <16 x double> @llvm.minnum.v16f64(<16 x double> %a, <16 x double> %b) #0
  store <16 x double> %val, ptr addrspace(1) %out, align 128
  ret void
}

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind "denormal-fp-math"="ieee,ieee" }
attributes #2 = { nounwind "denormal-fp-math"="preserve-sign,preserve-sign" }
