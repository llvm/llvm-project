; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI %s

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare { float, i1 } @llvm.amdgcn.div.scale.f32(float, float, i1) #1
declare { double, i1 } @llvm.amdgcn.div.scale.f64(double, double, i1) #1
declare float @llvm.fabs.f32(float) #1

; SI-LABEL: {{^}}test_div_scale_f32_1:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_1(float addrspace(1)* %out, float addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_2:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_2(float addrspace(1)* %out, float addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_1:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8
; SI: v_div_scale_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], [[A]]
; SI: buffer_store_dwordx2 [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f64_1(double addrspace(1)* %out, double addrspace(1)* %aptr, double addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0, align 8
  %b = load volatile double, double addrspace(1)* %gep.1, align 8

  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 false) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_2:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:8
; SI: v_div_scale_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[A]]
; SI: buffer_store_dwordx2 [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f64_2(double addrspace(1)* %out, double addrspace(1)* %aptr, double addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr double, double addrspace(1)* %gep.0, i32 1

  %a = load volatile double, double addrspace(1)* %gep.0, align 8
  %b = load volatile double, double addrspace(1)* %gep.1, align 8

  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 true) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_scalar_num_1:
; SI-DAG: buffer_load_dword [[B:v[0-9]+]]
; SI-DAG: s_load_dword [[A:s[0-9]+]]
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_scalar_num_1(float addrspace(1)* %out, float addrspace(1)* %in, float %a) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr float, float addrspace(1)* %in, i32 %tid

  %b = load float, float addrspace(1)* %gep, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_scalar_num_2:
; SI-DAG: buffer_load_dword [[B:v[0-9]+]]
; SI-DAG: s_load_dword [[A:s[0-9]+]]
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_scalar_num_2(float addrspace(1)* %out, float addrspace(1)* %in, float %a) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr float, float addrspace(1)* %in, i32 %tid

  %b = load float, float addrspace(1)* %gep, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_scalar_den_1:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]]
; SI-DAG: s_load_dword [[B:s[0-9]+]]
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_scalar_den_1(float addrspace(1)* %out, float addrspace(1)* %in, float %b) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr float, float addrspace(1)* %in, i32 %tid

  %a = load float, float addrspace(1)* %gep, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_scalar_den_2:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]]
; SI-DAG: s_load_dword [[B:s[0-9]+]]
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_scalar_den_2(float addrspace(1)* %out, float addrspace(1)* %in, float %b) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr float, float addrspace(1)* %in, i32 %tid

  %a = load float, float addrspace(1)* %gep, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_scalar_num_1:
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]]
; SI-DAG: s_load_dwordx2 [[A:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xd
; SI: v_div_scale_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], [[A]]
; SI: buffer_store_dwordx2 [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f64_scalar_num_1(double addrspace(1)* %out, double addrspace(1)* %in, double %a) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr double, double addrspace(1)* %in, i32 %tid

  %b = load double, double addrspace(1)* %gep, align 8

  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 false) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_scalar_num_2:
; SI-DAG: s_load_dwordx2 [[A:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: buffer_load_dwordx2 [[B:v\[[0-9]+:[0-9]+\]]]
; SI: v_div_scale_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[A]]
; SI: buffer_store_dwordx2 [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f64_scalar_num_2(double addrspace(1)* %out, double addrspace(1)* %in, double %a) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr double, double addrspace(1)* %in, i32 %tid

  %b = load double, double addrspace(1)* %gep, align 8

  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 true) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_scalar_den_1:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; SI-DAG: s_load_dwordx2 [[B:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xd
; SI: v_div_scale_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], [[A]]
; SI: buffer_store_dwordx2 [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f64_scalar_den_1(double addrspace(1)* %out, double addrspace(1)* %in, double %b) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr double, double addrspace(1)* %in, i32 %tid

  %a = load double, double addrspace(1)* %gep, align 8

  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 false) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_scalar_den_2:
; SI-DAG: buffer_load_dwordx2 [[A:v\[[0-9]+:[0-9]+\]]]
; SI-DAG: s_load_dwordx2 [[B:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0xd
; SI: v_div_scale_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], [[B]], [[A]]
; SI: buffer_store_dwordx2 [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f64_scalar_den_2(double addrspace(1)* %out, double addrspace(1)* %in, double %b) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep = getelementptr double, double addrspace(1)* %in, i32 %tid

  %a = load double, double addrspace(1)* %gep, align 8

  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 true) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_all_scalar_1:
; SI-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0x13
; SI-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0x1c
; SI: v_mov_b32_e32 [[VA:v[0-9]+]], [[A]]
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], [[VA]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_all_scalar_1(float addrspace(1)* %out, [8 x i32], float %a, [8 x i32], float %b) nounwind {
  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_all_scalar_2:
; SI-DAG: s_load_dword [[A:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0x13
; SI-DAG: s_load_dword [[B:s[0-9]+]], {{s\[[0-9]+:[0-9]+\]}}, 0x1c
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[B]]
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], [[VB]], [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_all_scalar_2(float addrspace(1)* %out, [8 x i32], float %a, [8 x i32], float %b) nounwind {
  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b, i1 true) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_all_scalar_1:
; SI-DAG: s_load_dwordx2 s{{\[}}[[A_LO:[0-9]+]]:[[A_HI:[0-9]+]]{{\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x13
; SI-DAG: s_load_dwordx2 [[B:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x1d
; SI-DAG: v_mov_b32_e32 v[[VA_LO:[0-9]+]], s[[A_LO]]
; SI-DAG: v_mov_b32_e32 v[[VA_HI:[0-9]+]], s[[A_HI]]
; SI: v_div_scale_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], v{{\[}}[[VA_LO]]:[[VA_HI]]{{\]}}
; SI: buffer_store_dwordx2 [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f64_all_scalar_1(double addrspace(1)* %out, [8 x i32], double %a, [8 x i32], double %b) nounwind {
  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 false) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_all_scalar_2:
; SI-DAG: s_load_dwordx2 [[A:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, 0x13
; SI-DAG: s_load_dwordx2 s{{\[}}[[B_LO:[0-9]+]]:[[B_HI:[0-9]+]]{{\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0x1d
; SI-DAG: v_mov_b32_e32 v[[VB_LO:[0-9]+]], s[[B_LO]]
; SI-DAG: v_mov_b32_e32 v[[VB_HI:[0-9]+]], s[[B_HI]]
; SI: v_div_scale_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], v{{\[}}[[VB_LO]]:[[VB_HI]]{{\]}}, [[A]]
; SI: buffer_store_dwordx2 [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f64_all_scalar_2(double addrspace(1)* %out, [8 x i32], double %a, [8 x i32], double %b) nounwind {
  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double %a, double %b, i1 true) nounwind readnone
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_inline_imm_num:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[A]], [[A]], 1.0
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_inline_imm_num(float addrspace(1)* %out, float addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %a = load float, float addrspace(1)* %gep.0, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float 1.0, float %a, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_inline_imm_den:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], 2.0, 2.0, [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_inline_imm_den(float addrspace(1)* %out, float addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %a = load float, float addrspace(1)* %gep.0, align 4

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float 2.0, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_fabs_num:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_and_b32_e32 [[ABS_A:v[0-9]+]], 0x7fffffff, [[A]]
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[B]], [[B]], [[ABS_A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_fabs_num(float addrspace(1)* %out, float addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %a.fabs = call float @llvm.fabs.f32(float %a) nounwind readnone

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a.fabs, float %b, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_fabs_den:
; SI-DAG: buffer_load_dword [[A:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64
; SI-DAG: buffer_load_dword [[B:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_and_b32_e32 [[ABS_B:v[0-9]+]], 0x7fffffff, [[B]]
; SI: v_div_scale_f32 [[RESULT0:v[0-9]+]], [[RESULT1:s\[[0-9]+:[0-9]+\]]], [[ABS_B]], [[ABS_B]], [[A]]
; SI: buffer_store_dword [[RESULT0]]
; SI: s_endpgm
define amdgpu_kernel void @test_div_scale_f32_fabs_den(float addrspace(1)* %out, float addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1

  %a = load volatile float, float addrspace(1)* %gep.0, align 4
  %b = load volatile float, float addrspace(1)* %gep.1, align 4

  %b.fabs = call float @llvm.fabs.f32(float %b) nounwind readnone

  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float %a, float %b.fabs, i1 false) nounwind readnone
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_val_undef_val:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; SI: v_div_scale_f32 v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, [[K]], v{{[0-9]+}}, [[K]]
define amdgpu_kernel void @test_div_scale_f32_val_undef_val(float addrspace(1)* %out) #0 {
  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float 8.0, float undef, i1 false)
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_undef_val_val:
; SI: s_mov_b32 [[K:s[0-9]+]], 0x41000000
; SI: v_div_scale_f32 v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, [[K]], [[K]], v{{[0-9]+}}
define amdgpu_kernel void @test_div_scale_f32_undef_val_val(float addrspace(1)* %out) #0 {
  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float undef, float 8.0, i1 false)
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f32_undef_undef_val:
; SI-NOT: v0
; SI: v_div_scale_f32 v{{[0-9]+}}, s{{\[[0-9]+:[0-9]+\]}}, v0, v0, v0
define amdgpu_kernel void @test_div_scale_f32_undef_undef_val(float addrspace(1)* %out) #0 {
  %result = call { float, i1 } @llvm.amdgcn.div.scale.f32(float undef, float undef, i1 false)
  %result0 = extractvalue { float, i1 } %result, 0
  store float %result0, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_div_scale_f64_val_undef_val:
; SI-DAG: s_mov_b32 s[[K_LO:[0-9]+]], 0{{$}}
; SI-DAG: s_mov_b32 s[[K_HI:[0-9]+]], 0x40200000
; SI: v_div_scale_f64 v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}, v[0:1], s{{\[}}[[K_LO]]:[[K_HI]]{{\]}}
define amdgpu_kernel void @test_div_scale_f64_val_undef_val(double addrspace(1)* %out) #0 {
  %result = call { double, i1 } @llvm.amdgcn.div.scale.f64(double 8.0, double undef, i1 false)
  %result0 = extractvalue { double, i1 } %result, 0
  store double %result0, double addrspace(1)* %out, align 8
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable }
