; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=SI-SAFE,GCN,FUNC %s
; RUN: llc -enable-no-nans-fp-math -enable-no-signed-zeros-fp-math -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=SI-NONAN,GCN-NONAN,GCN,FUNC %s

; RUN: llc -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=VI-SAFE,GCN,FUNC %s
; RUN: llc -enable-no-nans-fp-math -enable-no-signed-zeros-fp-math -march=amdgcn -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=VI-NONAN,GCN-NONAN,GCN,FUNC %s

; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -enable-var-scope --check-prefixes=EG,FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() #1

; The two inputs to the instruction are different SGPRs from the same
; super register, so we can't fold both SGPR operands even though they
; are both the same register.

; FUNC-LABEL: {{^}}s_test_fmin_legacy_subreg_inputs_f32:
; EG: MIN *
; SI-SAFE: v_min_legacy_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}

; SI-NONAN: v_min_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}

; VI-SAFE: v_cmp_nlt_f32_e32 vcc, s{{[0-9]+}}, v{{[0-9]+}}

; VI-NONAN: v_min_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_fmin_legacy_subreg_inputs_f32(float addrspace(1)* %out, <4 x float> %reg0) #0 {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = extractelement <4 x float> %reg0, i32 1
   %r2 = fcmp uge float %r0, %r1
   %r3 = select i1 %r2, float %r1, float %r0
   store float %r3, float addrspace(1)* %out
   ret void
}

; FUNC-LABEL: {{^}}s_test_fmin_legacy_ule_f32:
; GCN-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}

; SI-SAFE: v_mov_b32_e32 [[VA:v[0-9]+]], s[[#LOAD + 2]]

; GCN-NONAN: v_mov_b32_e32 [[VB:v[0-9]+]], s[[#LOAD + 3]]

; VI-SAFE: v_mov_b32_e32 [[VB:v[0-9]+]], s[[#LOAD + 3]]

; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, s[[#LOAD + 3]], [[VA]]

; VI-SAFE: v_mov_b32_e32 [[VA:v[0-9]+]], s[[#LOAD + 2]]
; VI-SAFE: v_cmp_ngt_f32_e32 vcc, s[[#LOAD + 2]], [[VB]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[VB]], [[VA]]

; GCN-NONAN: v_min_f32_e32 {{v[0-9]+}}, s[[#LOAD + 2]], [[VB]]
define amdgpu_kernel void @s_test_fmin_legacy_ule_f32(float addrspace(1)* %out, float %a, float %b) #0 {
  %cmp = fcmp ule float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; Nsz also needed
; FIXME: Should separate tests
; GCN-LABEL: {{^}}s_test_fmin_legacy_ule_f32_nnan_src:
; GCN: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}

; GCN-DAG: v_add_f32_e64 [[ADD_A:v[0-9]+]], s[[#LOAD + 2]], 1.0
; GCN-DAG: v_add_f32_e64 [[ADD_B:v[0-9]+]], s[[#LOAD + 3]], 2.0

; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[ADD_B]], [[ADD_A]]

; VI-SAFE: v_cmp_ngt_f32_e32 vcc, [[ADD_A]], [[ADD_B]]
; VI-SAFE: v_cndmask_b32_e32 {{v[0-9]+}}, [[ADD_B]], [[ADD_A]], vcc

; GCN-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[ADD_A]], [[ADD_B]]
define amdgpu_kernel void @s_test_fmin_legacy_ule_f32_nnan_src(float addrspace(1)* %out, float %a, float %b) #0 {
  %a.nnan = fadd nnan float %a, 1.0
  %b.nnan = fadd nnan float %b, 2.0
  %cmp = fcmp ule float %a.nnan, %b.nnan
  %val = select i1 %cmp, float %a.nnan, float %b.nnan
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ule_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]

; VI-SAFE: v_cmp_ngt_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]

; GCN-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_ule_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ule float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ole_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]

; VI-SAFE: v_cmp_le_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]

; GCN-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_ole_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ole float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_olt_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]

; VI-SAFE: v_cmp_lt_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]

; GCN-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_olt_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp olt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ult_f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]

; VI-SAFE: v_cmp_nge_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]

; GCN-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_ult_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ult float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ult_v1f32:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]

; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]

; VI-SAFE: v_cmp_nge_f32_e32 vcc, [[A]], [[B]]
; VI-SAFE: v_cndmask_b32_e32 v{{[0-9]+}}, [[B]], [[A]]

; GCN-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_ult_v1f32(<1 x float> addrspace(1)* %out, <1 x float> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr <1 x float>, <1 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <1 x float>, <1 x float> addrspace(1)* %gep.0, i32 1

  %a = load volatile <1 x float>, <1 x float> addrspace(1)* %gep.0
  %b = load volatile <1 x float>, <1 x float> addrspace(1)* %gep.1

  %cmp = fcmp ult <1 x float> %a, %b
  %val = select <1 x i1> %cmp, <1 x float> %a, <1 x float> %b
  store <1 x float> %val, <1 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ult_v2f32:
; GCN: {{buffer|flat}}_load_dwordx2
; GCN: {{buffer|flat}}_load_dwordx2
; SI-SAFE: v_min_legacy_f32_e32
; SI-SAFE: v_min_legacy_f32_e32

; VI-SAFE: v_cmp_nge_f32_e32
; VI-SAFE: v_cndmask_b32_e32
; VI-SAFE: v_cmp_nge_f32_e32
; VI-SAFE: v_cndmask_b32_e32

; GCN-NONAN: v_min_f32_e32
; GCN-NONAN: v_min_f32_e32
define amdgpu_kernel void @test_fmin_legacy_ult_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr <2 x float>, <2 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <2 x float>, <2 x float> addrspace(1)* %gep.0, i32 1

  %a = load volatile <2 x float>, <2 x float> addrspace(1)* %gep.0
  %b = load volatile <2 x float>, <2 x float> addrspace(1)* %gep.1

  %cmp = fcmp ult <2 x float> %a, %b
  %val = select <2 x i1> %cmp, <2 x float> %a, <2 x float> %b
  store <2 x float> %val, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ult_v3f32:
; SI-SAFE: v_min_legacy_f32_e32
; SI-SAFE: v_min_legacy_f32_e32
; SI-SAFE: v_min_legacy_f32_e32
; SI-SAFE-NOT: v_min_

; VI-SAFE: v_cmp_nge_f32_e32
; VI-SAFE: v_cndmask_b32_e32
; VI-SAFE: v_cmp_nge_f32_e32
; VI-SAFE: v_cndmask_b32_e32
; VI-SAFE: v_cmp_nge_f32_e32
; VI-SAFE: v_cndmask_b32_e32
; VI-NOT: v_cmp
; VI-NOT: v_cndmask

; GCN-NONAN: v_min_f32_e32
; GCN-NONAN: v_min_f32_e32
; GCN-NONAN: v_min_f32_e32
; GCN-NONAN-NOT: v_min_
define amdgpu_kernel void @test_fmin_legacy_ult_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr <3 x float>, <3 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <3 x float>, <3 x float> addrspace(1)* %gep.0, i32 1

  %a = load <3 x float>, <3 x float> addrspace(1)* %gep.0
  %b = load <3 x float>, <3 x float> addrspace(1)* %gep.1

  %cmp = fcmp ult <3 x float> %a, %b
  %val = select <3 x i1> %cmp, <3 x float> %a, <3 x float> %b
  store <3 x float> %val, <3 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ole_f32_multi_use:
; GCN: {{buffer|flat}}_load_dword [[A:v[0-9]+]]
; GCN: {{buffer|flat}}_load_dword [[B:v[0-9]+]]
; GCN-NOT: v_min
; GCN: v_cmp_le_f32
; GCN-NEXT: v_cndmask_b32
; GCN-NOT: v_min
; GCN: s_endpgm
define amdgpu_kernel void @test_fmin_legacy_ole_f32_multi_use(float addrspace(1)* %out0, i1 addrspace(1)* %out1, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ole float %a, %b
  %val0 = select i1 %cmp, float %a, float %b
  store float %val0, float addrspace(1)* %out0, align 4
  store i1 %cmp, i1 addrspace(1)* %out1
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
