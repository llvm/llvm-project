; RUN: llc -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=SI-SAFE,GCN,FUNC %s
; RUN: llc -enable-no-nans-fp-math -enable-no-signed-zeros-fp-math -mtriple=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN-NONAN,GCN,FUNC %s

; RUN: llc -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=VI-SAFE,GCN,FUNC %s
; RUN: llc -enable-no-nans-fp-math -enable-no-signed-zeros-fp-math -mtriple=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=GCN-NONAN,GCN,FUNC %s

; RUN: llc -mtriple=r600 -mcpu=redwood < %s | FileCheck -enable-var-scope --check-prefixes=EG,FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() #1

; FUNC-LABEL: {{^}}test_fmax_legacy_uge_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]

; VI-SAFE: v_cmp_nlt_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]

; GCN-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[A]], [[B]]

; EG: MAX
define amdgpu_kernel void @test_fmax_legacy_uge_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %cmp = fcmp uge float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_uge_f32_nnan_src:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-DAG: v_add_f32_e32 [[ADD_A:v[0-9]+]], 1.0, [[A]]
; GCN-DAG: v_add_f32_e32 [[ADD_B:v[0-9]+]], 2.0, [[B]]

; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[ADD_B]], [[ADD_A]]

; VI-SAFE: v_cmp_nlt_f32_e32 vcc, [[ADD_A]], [[ADD_B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[ADD_B]], [[ADD_A]]

; GCN-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[ADD_A]], [[ADD_B]]

; EG: MAX
define amdgpu_kernel void @test_fmax_legacy_uge_f32_nnan_src(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4
  %a.nnan = fadd nnan float %a, 1.0
  %b.nnan = fadd nnan float %b, 2.0

  %cmp = fcmp uge float %a.nnan, %b.nnan
  %val = select i1 %cmp, float %a.nnan, float %b.nnan
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_oge_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]

; VI-SAFE: v_cmp_ge_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]

; GCN-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; EG: MAX
define amdgpu_kernel void @test_fmax_legacy_oge_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %cmp = fcmp oge float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_ugt_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]

; VI-SAFE: v_cmp_nle_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]


; GCN-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; EG: MAX
define amdgpu_kernel void @test_fmax_legacy_ugt_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %cmp = fcmp ugt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_ogt_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]

; VI-SAFE: v_cmp_gt_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]

; GCN-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; EG: MAX
define amdgpu_kernel void @test_fmax_legacy_ogt_f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %cmp = fcmp ogt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, ptr addrspace(1) %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_ogt_v1f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]

; VI-SAFE: v_cmp_gt_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]


; GCN-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; EG: MAX
define amdgpu_kernel void @test_fmax_legacy_ogt_v1f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr <1 x float>, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr <1 x float>, ptr addrspace(1) %gep.0, i32 1

  %a = load volatile <1 x float>, ptr addrspace(1) %gep.0
  %b = load volatile <1 x float>, ptr addrspace(1) %gep.1

  %cmp = fcmp ogt <1 x float> %a, %b
  %val = select <1 x i1> %cmp, <1 x float> %a, <1 x float> %b
  store <1 x float> %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_ogt_v3f32:
; SI-SAFE: v_max_legacy_f32_e32
; SI-SAFE: v_max_legacy_f32_e32
; SI-SAFE: v_max_legacy_f32_e32

; VI-SAFE: v_cmp_gt_f32_e32
; VI-SAFE: v_cndmask_b32_e32
; VI-SAFE: v_cmp_gt_f32_e32
; VI-SAFE: v_cndmask_b32_e32
; VI-SAFE: v_cmp_gt_f32_e32
; VI-SAFE: v_cndmask_b32_e32
; VI-SAFE-NOT: v_cmp
; VI-SAFE-NOT: v_cndmask

; GCN-NONAN: v_max_f32_e32
; GCN-NONAN: v_max_f32_e32
; GCN-NONAN: v_max_f32_e32

; GCN-NOT: v_max
define amdgpu_kernel void @test_fmax_legacy_ogt_v3f32(ptr addrspace(1) %out, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr <3 x float>, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr <3 x float>, ptr addrspace(1) %gep.0, i32 1

  %a = load <3 x float>, ptr addrspace(1) %gep.0
  %b = load <3 x float>, ptr addrspace(1) %gep.1

  %cmp = fcmp ogt <3 x float> %a, %b
  %val = select <3 x i1> %cmp, <3 x float> %a, <3 x float> %b
  store <3 x float> %val, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_ogt_f32_multi_use:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-NOT: v_max_
; GCN: v_cmp_gt_f32
; GCN-NEXT: v_cndmask_b32
; GCN-NOT: v_max_

; EG: MAX
define amdgpu_kernel void @test_fmax_legacy_ogt_f32_multi_use(ptr addrspace(1) %out0, ptr addrspace(1) %out1, ptr addrspace(1) %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, ptr addrspace(1) %in, i32 %tid
  %gep.1 = getelementptr float, ptr addrspace(1) %gep.0, i32 1

  %a = load volatile float, ptr addrspace(1) %gep.0, align 4
  %b = load volatile float, ptr addrspace(1) %gep.1, align 4

  %cmp = fcmp ogt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, ptr addrspace(1) %out0, align 4
  store i1 %cmp, ptr addrspace(1) %out1
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
