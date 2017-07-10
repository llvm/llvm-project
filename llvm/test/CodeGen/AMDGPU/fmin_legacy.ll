; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI-SAFE -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -enable-no-nans-fp-math -enable-unsafe-fp-math  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI-NONAN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FIXME: Should replace unsafe-fp-math with no signed zeros.

declare i32 @llvm.r600.read.tidig.x() #1

; The two inputs to the instruction are different SGPRs from the same
; super register, so we can't fold both SGPR operands even though they
; are both the same register.

; FUNC-LABEL: {{^}}s_test_fmin_legacy_subreg_inputs_f32:
; EG: MIN *
; SI-SAFE: v_min_legacy_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
; SI-NONAN: v_min_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @s_test_fmin_legacy_subreg_inputs_f32(<4 x float> addrspace(1)* %out, <4 x float> inreg %reg0) #0 {
   %r0 = extractelement <4 x float> %reg0, i32 0
   %r1 = extractelement <4 x float> %reg0, i32 1
   %r2 = fcmp uge float %r0, %r1
   %r3 = select i1 %r2, float %r1, float %r0
   %vec = insertelement <4 x float> undef, float %r3, i32 0
   store <4 x float> %vec, <4 x float> addrspace(1)* %out, align 16
   ret void
}

; FUNC-LABEL: {{^}}s_test_fmin_legacy_ule_f32:
; SI-DAG: s_load_dword [[A:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[B:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc

; SI-SAFE-DAG: v_mov_b32_e32 [[VA:v[0-9]+]], [[A]]
; SI-NONAN-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[B]]

; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[VA]]
; SI-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[VB]]

define amdgpu_kernel void @s_test_fmin_legacy_ule_f32(float addrspace(1)* %out, float %a, float %b) #0 {
  %cmp = fcmp ule float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_ule_f32
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; SI-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_ule_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ule float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_ole_f32
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; SI-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_ole_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ole float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_olt_f32
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; SI-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_olt_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp olt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_ult_f32
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; SI-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_ult_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
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
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_min_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; SI-NONAN: v_min_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
define amdgpu_kernel void @test_fmin_legacy_ult_v1f32(<1 x float> addrspace(1)* %out, <1 x float> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr <1 x float>, <1 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <1 x float>, <1 x float> addrspace(1)* %gep.0, i32 1

  %a = load <1 x float>, <1 x float> addrspace(1)* %gep.0
  %b = load <1 x float>, <1 x float> addrspace(1)* %gep.1

  %cmp = fcmp ult <1 x float> %a, %b
  %val = select <1 x i1> %cmp, <1 x float> %a, <1 x float> %b
  store <1 x float> %val, <1 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ult_v2f32:
; SI: buffer_load_dwordx2
; SI: buffer_load_dwordx2
; SI-SAFE: v_min_legacy_f32_e32
; SI-SAFE: v_min_legacy_f32_e32

; SI-NONAN: v_min_f32_e32
; SI-NONAN: v_min_f32_e32
define amdgpu_kernel void @test_fmin_legacy_ult_v2f32(<2 x float> addrspace(1)* %out, <2 x float> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr <2 x float>, <2 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <2 x float>, <2 x float> addrspace(1)* %gep.0, i32 1

  %a = load <2 x float>, <2 x float> addrspace(1)* %gep.0
  %b = load <2 x float>, <2 x float> addrspace(1)* %gep.1

  %cmp = fcmp ult <2 x float> %a, %b
  %val = select <2 x i1> %cmp, <2 x float> %a, <2 x float> %b
  store <2 x float> %val, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_fmin_legacy_ult_v3f32:
; SI-SAFE: v_min_legacy_f32_e32
; SI-SAFE: v_min_legacy_f32_e32
; SI-SAFE: v_min_legacy_f32_e32

; SI-NONAN: v_min_f32_e32
; SI-NONAN: v_min_f32_e32
; SI-NONAN: v_min_f32_e32
define amdgpu_kernel void @test_fmin_legacy_ult_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr <3 x float>, <3 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <3 x float>, <3 x float> addrspace(1)* %gep.0, i32 1

  %a = load <3 x float>, <3 x float> addrspace(1)* %gep.0
  %b = load <3 x float>, <3 x float> addrspace(1)* %gep.1

  %cmp = fcmp ult <3 x float> %a, %b
  %val = select <3 x i1> %cmp, <3 x float> %a, <3 x float> %b
  store <3 x float> %val, <3 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @test_fmin_legacy_ole_f32_multi_use
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-NOT: v_min
; SI: v_cmp_le_f32
; SI-NEXT: v_cndmask_b32
; SI-NOT: v_min
; SI: s_endpgm
define amdgpu_kernel void @test_fmin_legacy_ole_f32_multi_use(float addrspace(1)* %out0, i1 addrspace(1)* %out1, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
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
