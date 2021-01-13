; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fabs.f32(float) #1
declare float @llvm.fma.f32(float, float, float) nounwind readnone

; FUNC-LABEL: @commute_add_imm_fabs_f32
; SI: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: v_add_f32_e64 [[REG:v[0-9]+]], |[[X]]|, 2.0
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_add_imm_fabs_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %x = load float, float addrspace(1)* %gep.0
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %z = fadd float 2.0, %x.fabs
  store float %z, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @commute_mul_imm_fneg_fabs_f32
; SI: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: v_mul_f32_e64 [[REG:v[0-9]+]], |[[X]]|, -4.0
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_mul_imm_fneg_fabs_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %x = load float, float addrspace(1)* %gep.0
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %x.fneg.fabs = fsub float -0.000000e+00, %x.fabs
  %z = fmul float 4.0, %x.fneg.fabs
  store float %z, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @commute_mul_imm_fneg_f32
; SI: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: v_mul_f32_e32 [[REG:v[0-9]+]], -4.0, [[X]]
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_mul_imm_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %x = load float, float addrspace(1)* %gep.0
  %x.fneg = fsub float -0.000000e+00, %x
  %z = fmul float 4.0, %x.fneg
  store float %z, float addrspace(1)* %out
  ret void
}

; FIXME: Should use SGPR for literal.
; FUNC-LABEL: @commute_add_lit_fabs_f32
; SI: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64{{$}}
; SI: s_mov_b32 [[K:s[0-9]+]], 0x44800000
; SI: v_add_f32_e64 [[REG:v[0-9]+]], |[[X]]|, [[K]]
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_add_lit_fabs_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %x = load float, float addrspace(1)* %gep.0
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %z = fadd float 1024.0, %x.fabs
  store float %z, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @commute_add_fabs_f32
; SI-DAG: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[Y:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_add_f32_e64 [[REG:v[0-9]+]], [[X]], |[[Y]]|
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_add_fabs_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %y.fabs = call float @llvm.fabs.f32(float %y) #1
  %z = fadd float %x, %y.fabs
  store float %z, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @commute_mul_fneg_f32
; SI-DAG: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[Y:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_mul_f32_e64 [[REG:v[0-9]+]], [[X]], -[[Y]]
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_mul_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %y.fneg = fsub float -0.000000e+00, %y
  %z = fmul float %x, %y.fneg
  store float %z, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @commute_mul_fabs_fneg_f32
; SI-DAG: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[Y:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_mul_f32_e64 [[REG:v[0-9]+]], [[X]], -|[[Y]]|
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_mul_fabs_fneg_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %y.fabs = call float @llvm.fabs.f32(float %y) #1
  %y.fabs.fneg = fsub float -0.000000e+00, %y.fabs
  %z = fmul float %x, %y.fabs.fneg
  store float %z, float addrspace(1)* %out
  ret void
}

; There's no reason to commute this.
; FUNC-LABEL: @commute_mul_fabs_x_fabs_y_f32
; SI-DAG: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[Y:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_mul_f32_e64 [[REG:v[0-9]+]], |[[X]]|, |[[Y]]|
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_mul_fabs_x_fabs_y_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %y.fabs = call float @llvm.fabs.f32(float %y) #1
  %z = fmul float %x.fabs, %y.fabs
  store float %z, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: @commute_mul_fabs_x_fneg_fabs_y_f32
; SI-DAG: buffer_load_dword [[X:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[Y:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_mul_f32_e64 [[REG:v[0-9]+]], |[[X]]|, -|[[Y]]|
; SI: buffer_store_dword [[REG]]
define amdgpu_kernel void @commute_mul_fabs_x_fneg_fabs_y_f32(float addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.0 = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %x = load volatile float, float addrspace(1)* %gep.0
  %y = load volatile float, float addrspace(1)* %gep.1
  %x.fabs = call float @llvm.fabs.f32(float %x) #1
  %y.fabs = call float @llvm.fabs.f32(float %y) #1
  %y.fabs.fneg = fsub float -0.000000e+00, %y.fabs
  %z = fmul float %x.fabs, %y.fabs.fneg
  store float %z, float addrspace(1)* %out
  ret void
}

; Make sure we commute the multiply part for the constant in src0 even
; though we have negate modifier on src2.

; SI-LABEL: {{^}}fma_a_2.0_neg_b_f32
; SI-DAG: buffer_load_dword [[R1:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 glc{{$}}
; SI-DAG: buffer_load_dword [[R2:v[0-9]+]], {{v\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 0 addr64 offset:4
; SI: v_fma_f32 [[RESULT:v[0-9]+]], [[R1]], 2.0, |[[R2]]|
; SI: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @fma_a_2.0_neg_b_f32(float addrspace(1)* %out, float addrspace(1)* %in) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.0 = getelementptr float, float addrspace(1)* %out, i32 %tid
  %gep.1 = getelementptr float, float addrspace(1)* %gep.0, i32 1
  %gep.out = getelementptr float, float addrspace(1)* %out, i32 %tid

  %r1 = load volatile float, float addrspace(1)* %gep.0
  %r2 = load volatile float, float addrspace(1)* %gep.1

  %r2.fabs = call float @llvm.fabs.f32(float %r2)

  %r3 = tail call float @llvm.fma.f32(float %r1, float 2.0, float %r2.fabs)
  store float %r3, float addrspace(1)* %gep.out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
