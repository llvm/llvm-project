; RUN: llc -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI -check-prefix=SI-SAFE -check-prefix=FUNC %s
; RUN: llc -enable-no-nans-fp-math -enable-unsafe-fp-math -march=amdgcn -mcpu=SI < %s | FileCheck -check-prefix=SI-NONAN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FIXME: Should replace unsafe-fp-math with no signed zeros.

declare i32 @llvm.r600.read.tidig.x() #1

; FUNC-LABEL: @test_fmax_legacy_uge_f32
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; SI-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[B]], [[A]]

; EG: MAX
define void @test_fmax_legacy_uge_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load float, float addrspace(1)* %gep.0, align 4
  %b = load float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp uge float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmax_legacy_oge_f32
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; SI-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; EG: MAX
define void @test_fmax_legacy_oge_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load float, float addrspace(1)* %gep.0, align 4
  %b = load float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp oge float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmax_legacy_ugt_f32
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; SI-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; EG: MAX
define void @test_fmax_legacy_ugt_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load float, float addrspace(1)* %gep.0, align 4
  %b = load float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ugt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: @test_fmax_legacy_ogt_f32
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; SI-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; EG: MAX
define void @test_fmax_legacy_ogt_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load float, float addrspace(1)* %gep.0, align 4
  %b = load float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ogt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_ogt_v1f32:
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-SAFE: v_max_legacy_f32_e32 {{v[0-9]+}}, [[A]], [[B]]
; SI-NONAN: v_max_f32_e32 {{v[0-9]+}}, [[B]], [[A]]
; EG: MAX
define void @test_fmax_legacy_ogt_v1f32(<1 x float> addrspace(1)* %out, <1 x float> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr <1 x float>, <1 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <1 x float>, <1 x float> addrspace(1)* %gep.0, i32 1

  %a = load <1 x float>, <1 x float> addrspace(1)* %gep.0
  %b = load <1 x float>, <1 x float> addrspace(1)* %gep.1

  %cmp = fcmp ogt <1 x float> %a, %b
  %val = select <1 x i1> %cmp, <1 x float> %a, <1 x float> %b
  store <1 x float> %val, <1 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_fmax_legacy_ogt_v3f32:
; SI-SAFE: v_max_legacy_f32_e32
; SI-SAFE: v_max_legacy_f32_e32
; SI-SAFE: v_max_legacy_f32_e32
; SI-NONAN: v_max_f32_e32
; SI-NONAN: v_max_f32_e32
; SI-NONAN: v_max_f32_e32
define void @test_fmax_legacy_ogt_v3f32(<3 x float> addrspace(1)* %out, <3 x float> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr <3 x float>, <3 x float> addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr <3 x float>, <3 x float> addrspace(1)* %gep.0, i32 1

  %a = load <3 x float>, <3 x float> addrspace(1)* %gep.0
  %b = load <3 x float>, <3 x float> addrspace(1)* %gep.1

  %cmp = fcmp ogt <3 x float> %a, %b
  %val = select <3 x i1> %cmp, <3 x float> %a, <3 x float> %b
  store <3 x float> %val, <3 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @test_fmax_legacy_ogt_f32_multi_use
; SI: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI-NOT: v_max_
; SI: v_cmp_gt_f32
; SI-NEXT: v_cndmask_b32
; SI-NOT: v_max_

; EG: MAX
define void @test_fmax_legacy_ogt_f32_multi_use(float addrspace(1)* %out0, i1 addrspace(1)* %out1, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load float, float addrspace(1)* %gep.0, align 4
  %b = load float, float addrspace(1)* %gep.1, align 4

  %cmp = fcmp ogt float %a, %b
  %val = select i1 %cmp, float %a, float %b
  store float %val, float addrspace(1)* %out0, align 4
  store i1 %cmp, i1addrspace(1)* %out1
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
