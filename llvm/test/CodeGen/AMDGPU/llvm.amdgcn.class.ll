; RUN:  llc -amdgpu-scalarize-global-loads=false  -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

declare i1 @llvm.amdgcn.class.f32(float, i32) #1
declare i1 @llvm.amdgcn.class.f64(double, i32) #1
declare i32 @llvm.amdgcn.workitem.id.x() #1
declare float @llvm.fabs.f32(float) #1
declare double @llvm.fabs.f64(double) #1

; SI-LABEL: {{^}}test_class_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f32_e32 vcc, [[SA]], [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %result = call i1 @llvm.amdgcn.class.f32(float %a, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fabs_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], |[[SA]]|, [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_fabs_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %a.fabs = call float @llvm.fabs.f32(float %a) #1
  %result = call i1 @llvm.amdgcn.class.f32(float %a.fabs, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fneg_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -[[SA]], [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_fneg_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %a.fneg = fsub float -0.0, %a
  %result = call i1 @llvm.amdgcn.class.f32(float %a.fneg, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fneg_fabs_f32:
; SI-DAG: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xc
; SI: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f32_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -|[[SA]]|, [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_fneg_fabs_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %a.fabs = call float @llvm.fabs.f32(float %a) #1
  %a.fneg.fabs = fsub float -0.0, %a.fabs
  %result = call i1 @llvm.amdgcn.class.f32(float %a.fneg.fabs, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_1_f32:
; SI: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_cmp_class_f32_e64 [[COND:s\[[0-9]+:[0-9]+\]]], [[SA]], 1{{$}}
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[COND]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_1_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f32(float %a, i32 1) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_64_f32:
; SI: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_cmp_class_f32_e64 [[COND:s\[[0-9]+:[0-9]+\]]], [[SA]], 64{{$}}
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[COND]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_64_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f32(float %a, i32 64) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; Set all 10 bits of mask
; SI-LABEL: {{^}}test_class_full_mask_f32:
; SI: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x3ff{{$}}
; SI: v_cmp_class_f32_e32 vcc, [[SA]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_full_mask_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f32(float %a, i32 1023) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_9bit_mask_f32:
; SI: s_load_dword [[SA:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1ff{{$}}
; SI: v_cmp_class_f32_e32 vcc, [[SA]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_9bit_mask_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f32(float %a, i32 511) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}v_test_class_full_mask_f32:
; SI-DAG: buffer_load_dword [[VA:v[0-9]+]]
; SI-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1ff{{$}}
; SI: v_cmp_class_f32_e32 vcc, [[VA]], [[MASK]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @v_test_class_full_mask_f32(i32 addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep.in

  %result = call i1 @llvm.amdgcn.class.f32(float %a, i32 511) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_inline_imm_constant_dynamic_mask_f32:
; SI-DAG: buffer_load_dword [[VB:v[0-9]+]]
; SI: v_cmp_class_f32_e32 vcc, 1.0, [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_inline_imm_constant_dynamic_mask_f32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %b = load i32, i32 addrspace(1)* %gep.in

  %result = call i1 @llvm.amdgcn.class.f32(float 1.0, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; FIXME: Why isn't this using a literal constant operand?
; SI-LABEL: {{^}}test_class_lit_constant_dynamic_mask_f32:
; SI-DAG: buffer_load_dword [[VB:v[0-9]+]]
; SI-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; SI: v_cmp_class_f32_e32 vcc, [[VK]], [[VB]]
; SI-NEXT: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_lit_constant_dynamic_mask_f32(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %b = load i32, i32 addrspace(1)* %gep.in

  %result = call i1 @llvm.amdgcn.class.f32(float 1024.0, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_f64:
; SI-DAG: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f64_e32 vcc, [[SA]], [[VB]]
; SI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_f64(i32 addrspace(1)* %out, double %a, i32 %b) #0 {
  %result = call i1 @llvm.amdgcn.class.f64(double %a, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fabs_f64:
; SI-DAG: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f64_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], |[[SA]]|, [[VB]]
; SI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_fabs_f64(i32 addrspace(1)* %out, double %a, i32 %b) #0 {
  %a.fabs = call double @llvm.fabs.f64(double %a) #1
  %result = call i1 @llvm.amdgcn.class.f64(double %a.fabs, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fneg_f64:
; SI-DAG: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f64_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -[[SA]], [[VB]]
; SI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_fneg_f64(i32 addrspace(1)* %out, double %a, i32 %b) #0 {
  %a.fneg = fsub double -0.0, %a
  %result = call i1 @llvm.amdgcn.class.f64(double %a.fneg, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_fneg_fabs_f64:
; SI-DAG: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI-DAG: s_load_dword [[SB:s[0-9]+]], s{{\[[0-9]+:[0-9]+\]}}, 0xd
; SI-DAG: v_mov_b32_e32 [[VB:v[0-9]+]], [[SB]]
; SI: v_cmp_class_f64_e64 [[CMP:s\[[0-9]+:[0-9]+\]]], -|[[SA]]|, [[VB]]
; SI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, [[CMP]]
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_fneg_fabs_f64(i32 addrspace(1)* %out, double %a, i32 %b) #0 {
  %a.fabs = call double @llvm.fabs.f64(double %a) #1
  %a.fneg.fabs = fsub double -0.0, %a.fabs
  %result = call i1 @llvm.amdgcn.class.f64(double %a.fneg.fabs, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_1_f64:
; SI: v_cmp_class_f64_e64 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 1{{$}}
; SI: s_endpgm
define amdgpu_kernel void @test_class_1_f64(i32 addrspace(1)* %out, double %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f64(double %a, i32 1) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_64_f64:
; SI: v_cmp_class_f64_e64 {{s\[[0-9]+:[0-9]+\]}}, {{s\[[0-9]+:[0-9]+\]}}, 64{{$}}
; SI: s_endpgm
define amdgpu_kernel void @test_class_64_f64(i32 addrspace(1)* %out, double %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f64(double %a, i32 64) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; Set all 9 bits of mask
; SI-LABEL: {{^}}test_class_full_mask_f64:
; SI: s_load_dwordx2 [[SA:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, 0xb
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1ff{{$}}
; SI: v_cmp_class_f64_e32 vcc, [[SA]], [[MASK]]
; SI-NOT: vcc
; SI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI-NEXT: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_full_mask_f64(i32 addrspace(1)* %out, double %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f64(double %a, i32 511) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}v_test_class_full_mask_f64:
; SI-DAG: buffer_load_dwordx2 [[VA:v\[[0-9]+:[0-9]+\]]]
; SI-DAG: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x1ff{{$}}
; SI: v_cmp_class_f64_e32 vcc, [[VA]], [[MASK]]
; SI-NOT: vcc
; SI: v_cndmask_b32_e64 [[RESULT:v[0-9]+]], 0, -1, vcc
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @v_test_class_full_mask_f64(i32 addrspace(1)* %out, double addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr double, double addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load double, double addrspace(1)* %in

  %result = call i1 @llvm.amdgcn.class.f64(double %a, i32 511) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_inline_imm_constant_dynamic_mask_f64:
; XSI: v_cmp_class_f64_e32 vcc, 1.0,
; SI: v_cmp_class_f64_e32 vcc,
; SI: s_endpgm
define amdgpu_kernel void @test_class_inline_imm_constant_dynamic_mask_f64(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %b = load i32, i32 addrspace(1)* %gep.in

  %result = call i1 @llvm.amdgcn.class.f64(double 1.0, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_lit_constant_dynamic_mask_f64:
; SI: v_cmp_class_f64_e32 vcc, s{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}
; SI: s_endpgm
define amdgpu_kernel void @test_class_lit_constant_dynamic_mask_f64(i32 addrspace(1)* %out, i32 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %b = load i32, i32 addrspace(1)* %gep.in

  %result = call i1 @llvm.amdgcn.class.f64(double 1024.0, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %gep.out, align 4
  ret void
}

; SI-LABEL: {{^}}test_fold_or_class_f32_0:
; SI-NOT: v_cmp_class
; SI: v_cmp_class_f32_e64 {{s\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, 3{{$}}
; SI-NOT: v_cmp_class
; SI: s_endpgm
define amdgpu_kernel void @test_fold_or_class_f32_0(i32 addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep.in

  %class0 = call i1 @llvm.amdgcn.class.f32(float %a, i32 1) #1
  %class1 = call i1 @llvm.amdgcn.class.f32(float %a, i32 3) #1
  %or = or i1 %class0, %class1

  %sext = sext i1 %or to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_fold_or3_class_f32_0:
; SI-NOT: v_cmp_class
; SI: v_cmp_class_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, 7{{$}}
; SI-NOT: v_cmp_class
; SI: s_endpgm
define amdgpu_kernel void @test_fold_or3_class_f32_0(i32 addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep.in

  %class0 = call i1 @llvm.amdgcn.class.f32(float %a, i32 1) #1
  %class1 = call i1 @llvm.amdgcn.class.f32(float %a, i32 2) #1
  %class2 = call i1 @llvm.amdgcn.class.f32(float %a, i32 4) #1
  %or.0 = or i1 %class0, %class1
  %or.1 = or i1 %or.0, %class2

  %sext = sext i1 %or.1 to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_fold_or_all_tests_class_f32_0:
; SI-NOT: v_cmp_class
; SI: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x3ff{{$}}
; SI: v_cmp_class_f32_e32 vcc, v{{[0-9]+}}, [[MASK]]{{$}}
; SI-NOT: v_cmp_class
; SI: s_endpgm
define amdgpu_kernel void @test_fold_or_all_tests_class_f32_0(i32 addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep.in

  %class0 = call i1 @llvm.amdgcn.class.f32(float %a, i32 1) #1
  %class1 = call i1 @llvm.amdgcn.class.f32(float %a, i32 2) #1
  %class2 = call i1 @llvm.amdgcn.class.f32(float %a, i32 4) #1
  %class3 = call i1 @llvm.amdgcn.class.f32(float %a, i32 8) #1
  %class4 = call i1 @llvm.amdgcn.class.f32(float %a, i32 16) #1
  %class5 = call i1 @llvm.amdgcn.class.f32(float %a, i32 32) #1
  %class6 = call i1 @llvm.amdgcn.class.f32(float %a, i32 64) #1
  %class7 = call i1 @llvm.amdgcn.class.f32(float %a, i32 128) #1
  %class8 = call i1 @llvm.amdgcn.class.f32(float %a, i32 256) #1
  %class9 = call i1 @llvm.amdgcn.class.f32(float %a, i32 512) #1
  %or.0 = or i1 %class0, %class1
  %or.1 = or i1 %or.0, %class2
  %or.2 = or i1 %or.1, %class3
  %or.3 = or i1 %or.2, %class4
  %or.4 = or i1 %or.3, %class5
  %or.5 = or i1 %or.4, %class6
  %or.6 = or i1 %or.5, %class7
  %or.7 = or i1 %or.6, %class8
  %or.8 = or i1 %or.7, %class9
  %sext = sext i1 %or.8 to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_fold_or_class_f32_1:
; SI-NOT: v_cmp_class
; SI: v_cmp_class_f32_e64 {{s\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, 12{{$}}
; SI-NOT: v_cmp_class
; SI: s_endpgm
define amdgpu_kernel void @test_fold_or_class_f32_1(i32 addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep.in

  %class0 = call i1 @llvm.amdgcn.class.f32(float %a, i32 4) #1
  %class1 = call i1 @llvm.amdgcn.class.f32(float %a, i32 8) #1
  %or = or i1 %class0, %class1

  %sext = sext i1 %or to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_fold_or_class_f32_2:
; SI-NOT: v_cmp_class
; SI: v_cmp_class_f32_e64 {{s\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, 7{{$}}
; SI-NOT: v_cmp_class
; SI: s_endpgm
define amdgpu_kernel void @test_fold_or_class_f32_2(i32 addrspace(1)* %out, float addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep.in

  %class0 = call i1 @llvm.amdgcn.class.f32(float %a, i32 7) #1
  %class1 = call i1 @llvm.amdgcn.class.f32(float %a, i32 7) #1
  %or = or i1 %class0, %class1

  %sext = sext i1 %or to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_no_fold_or_class_f32_0:
; SI-DAG: v_cmp_class_f32_e64 {{s\[[0-9]+:[0-9]+\]}}, v{{[0-9]+}}, 4{{$}}
; SI-DAG: v_cmp_class_f32_e64 {{s\[[0-9]+:[0-9]+\]}}, s{{[0-9]+}}, 8{{$}}
; SI: s_or_b64
; SI: s_endpgm
define amdgpu_kernel void @test_no_fold_or_class_f32_0(i32 addrspace(1)* %out, float addrspace(1)* %in, float %b) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %gep.in = getelementptr float, float addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %a = load float, float addrspace(1)* %gep.in

  %class0 = call i1 @llvm.amdgcn.class.f32(float %a, i32 4) #1
  %class1 = call i1 @llvm.amdgcn.class.f32(float %b, i32 8) #1
  %or = or i1 %class0, %class1

  %sext = sext i1 %or to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_0_f32:
; SI-NOT: v_cmp_class
; SI: v_mov_b32_e32 [[RESULT:v[0-9]+]], 0{{$}}
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_0_f32(i32 addrspace(1)* %out, float %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f32(float %a, i32 0) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}test_class_0_f64:
; SI-NOT: v_cmp_class
; SI: v_mov_b32_e32 [[RESULT:v[0-9]+]], 0{{$}}
; SI: buffer_store_dword [[RESULT]]
; SI: s_endpgm
define amdgpu_kernel void @test_class_0_f64(i32 addrspace(1)* %out, double %a) #0 {
  %result = call i1 @llvm.amdgcn.class.f64(double %a, i32 0) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

; FIXME: Why is the extension still here?
; SI-LABEL: {{^}}test_class_undef_f32:
; SI-NOT: v_cmp_class
; SI: v_cndmask_b32_e64 v{{[0-9]+}}, 0, -1,
; SI: buffer_store_dword
define amdgpu_kernel void @test_class_undef_f32(i32 addrspace(1)* %out, float %a, i32 %b) #0 {
  %result = call i1 @llvm.amdgcn.class.f32(float undef, i32 %b) #1
  %sext = sext i1 %result to i32
  store i32 %sext, i32 addrspace(1)* %out, align 4
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
