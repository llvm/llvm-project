; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=GCN %s

declare float @llvm.fma.f32(float, float, float) #1
declare double @llvm.fma.f64(double, double, double) #1
declare float @llvm.fmuladd.f32(float, float, float) #1
declare float @llvm.amdgcn.div.fixup.f32(float, float, float) #1


; GCN-LABEL: {{^}}test_sgpr_use_twice_binop:
; GCN: s_load_dword [[SGPR:s[0-9]+]],
; GCN: v_add_f32_e64 [[RESULT:v[0-9]+]], [[SGPR]], [[SGPR]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_twice_binop(float addrspace(1)* %out, float %a) #0 {
  %dbl = fadd float %a, %a
  store float %dbl, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_three_ternary_op:
; GCN: s_load_dword [[SGPR:s[0-9]+]],
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], [[SGPR]], [[SGPR]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_three_ternary_op(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %a, float %a) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_a_b:
; SI-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; VI-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GCN: v_mov_b32_e32 [[VGPR1:v[0-9]+]], s[[#LOAD + 3]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], s[[#LOAD + 2]], s[[#LOAD + 2]], [[VGPR1]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_twice_ternary_op_a_a_b(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %a, float %b) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_use_s_v_s:
; SI: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; SI: buffer_load_dword [[VA0:v[0-9]+]]
; SI-NEXT: s_waitcnt vmcnt(0)
; SI-NEXT: buffer_load_dword [[VA1:v[0-9]+]]
; SI-NEXT: s_waitcnt vmcnt(0)

; GCN-NOT: v_mov_b32

; VI: buffer_load_dword [[VA0:v[0-9]+]]
; VI-NEXT: s_waitcnt vmcnt(0)
; VI-NEXT: buffer_load_dword [[VA1:v[0-9]+]]
; VI-NEXT: s_waitcnt vmcnt(0)
; VI: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}

; GCN-NOT: v_mov_b32
; GCN: v_mov_b32_e32 [[VB:v[0-9]+]], s[[#LOAD + 3]]
; GCN-NOT: v_mov_b32

; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], s[[#LOAD + 2]], [[VA0]], [[VB]]
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], s[[#LOAD + 2]], [[VA1]], [[VB]]
; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
define amdgpu_kernel void @test_use_s_v_s(float addrspace(1)* %out, float %a, float %b, float addrspace(1)* %in) #0 {
  %va0 = load volatile float, float addrspace(1)* %in
  %va1 = load volatile float, float addrspace(1)* %in
  %fma0 = call float @llvm.fma.f32(float %a, float %va0, float %b) #1
  %fma1 = call float @llvm.fma.f32(float %a, float %va1, float %b) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_b_a:
; SI-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; VI-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GCN: v_mov_b32_e32 [[VGPR1:v[0-9]+]], s[[#LOAD + 3]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], s[[#LOAD + 2]], [[VGPR1]], s[[#LOAD + 2]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_twice_ternary_op_a_b_a(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %b, float %a) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_b_a_a:
; SI-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; VI-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GCN: v_mov_b32_e32 [[VGPR1:v[0-9]+]], s[[#LOAD + 3]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[VGPR1]], s[[#LOAD + 2]], s[[#LOAD + 2]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_twice_ternary_op_b_a_a(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma = call float @llvm.fma.f32(float %b, float %a, float %a) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_a_imm:
; GCN: s_load_dword [[SGPR:s[0-9]+]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], [[SGPR]], 2.0
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_twice_ternary_op_a_a_imm(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %a, float 2.0) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_imm_a:
; GCN: s_load_dword [[SGPR:s[0-9]+]]
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], 2.0, [[SGPR]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_twice_ternary_op_a_imm_a(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float 2.0, float %a) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; Don't use fma since fma c, x, y is canonicalized to fma x, c, y
; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_imm_a_a:
; GCN: s_load_dword [[SGPR:s[0-9]+]]
; GCN: v_div_fixup_f32 [[RESULT:v[0-9]+]], 2.0, [[SGPR]], [[SGPR]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_twice_ternary_op_imm_a_a(float addrspace(1)* %out, float %a) #0 {
  %val = call float @llvm.amdgcn.div.fixup.f32(float 2.0, float %a, float %a) #1
  store float %val, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_sgpr_use_twice_ternary_op_a_a_kimm:
; GCN-DAG: s_load_dword [[SGPR:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VK:v[0-9]+]], 0x44800000
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[SGPR]], [[SGPR]], [[VK]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_sgpr_use_twice_ternary_op_a_a_kimm(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float %a, float 1024.0) #1
  store float %fma, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_k_k_s:
; GCN-DAG: s_load_dword [[SGPR:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VGPR:v[0-9]+]], [[SGPR]]
; GCN-DAG: s_mov_b32 [[SK:s[0-9]+]], 0x44800000
; GCN: v_fma_f32 [[RESULT0:v[0-9]+]], [[SK]], [[SK]], [[VGPR]]
; GCN: buffer_store_dword [[RESULT0]]
define amdgpu_kernel void @test_literal_use_twice_ternary_op_k_k_s(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float 1024.0, float 1024.0, float %a) #1
  store float %fma, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_k_k_s_x2:
; GCN-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-DAG: v_mov_b32_e32 [[VGPR0:v[0-9]+]], s[[#LOAD + 2]]
; GCN-DAG: v_mov_b32_e32 [[VGPR1:v[0-9]+]], s[[#LOAD + 3]]
; GCN-DAG: s_mov_b32 [[SK:s[0-9]+]], 0x44800000
; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[SK]], [[SK]], [[VGPR0]]
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[SK]], [[SK]], [[VGPR1]]
; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
; GCN: s_endpgm
define amdgpu_kernel void @test_literal_use_twice_ternary_op_k_k_s_x2(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma0 = call float @llvm.fma.f32(float 1024.0, float 1024.0, float %a) #1
  %fma1 = call float @llvm.fma.f32(float 1024.0, float 1024.0, float %b) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_k_s_k:
; GCN-DAG: s_load_dword [[SGPR:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VGPR:v[0-9]+]], [[SGPR]]
; GCN-DAG: s_mov_b32 [[SK:s[0-9]+]], 0x44800000
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[VGPR]], [[SK]], [[SK]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_literal_use_twice_ternary_op_k_s_k(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float 1024.0, float %a, float 1024.0) #1
  store float %fma, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_k_s_k_x2:
; GCN-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-DAG: v_mov_b32_e32 [[VGPR0:v[0-9]+]], s[[#LOAD + 2]]
; GCN-DAG: v_mov_b32_e32 [[VGPR1:v[0-9]+]], s[[#LOAD + 3]]
; GCN-DAG: s_mov_b32 [[SK:s[0-9]+]], 0x44800000
; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[VGPR0]], [[SK]], [[SK]]
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[VGPR1]], [[SK]], [[SK]]
; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
; GCN: s_endpgm
define amdgpu_kernel void @test_literal_use_twice_ternary_op_k_s_k_x2(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma0 = call float @llvm.fma.f32(float 1024.0, float %a, float 1024.0) #1
  %fma1 = call float @llvm.fma.f32(float 1024.0, float %b, float 1024.0) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_s_k_k:
; GCN-DAG: s_load_dword [[SGPR:s[0-9]+]]
; GCN-DAG: v_mov_b32_e32 [[VGPR:v[0-9]+]], [[SGPR]]
; GCN-DAG: s_mov_b32 [[SK:s[0-9]+]], 0x44800000
; GCN: v_fma_f32 [[RESULT:v[0-9]+]], [[VGPR]], [[SK]], [[SK]]
; GCN: buffer_store_dword [[RESULT]]
define amdgpu_kernel void @test_literal_use_twice_ternary_op_s_k_k(float addrspace(1)* %out, float %a) #0 {
  %fma = call float @llvm.fma.f32(float %a, float 1024.0, float 1024.0) #1
  store float %fma, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_literal_use_twice_ternary_op_s_k_k_x2:
; GCN-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, {{0x9|0x24}}
; GCN-DAG: v_mov_b32_e32 [[VGPR0:v[0-9]+]], s[[#LOAD + 2]]
; GCN-DAG: v_mov_b32_e32 [[VGPR1:v[0-9]+]], s[[#LOAD + 3]]
; GCN-DAG: s_mov_b32 [[SK:s[0-9]+]], 0x44800000
; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], [[VGPR0]], [[SK]], [[SK]]
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], [[VGPR1]], [[SK]], [[SK]]
; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
; GCN: s_endpgm
define amdgpu_kernel void @test_literal_use_twice_ternary_op_s_k_k_x2(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma0 = call float @llvm.fma.f32(float %a, float 1024.0, float 1024.0) #1
  %fma1 = call float @llvm.fma.f32(float %b, float 1024.0, float 1024.0) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_s0_s1_k_f32:
; SI-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, 0x9
; VI-DAG: s_load_dwordx4 s[[[#LOAD:]]:{{[0-9]+}}], s{{\[[0-9]+:[0-9]+\]}}, 0x24
; GCN-DAG: v_mov_b32_e32 [[VK0:v[0-9]+]], 0x44800000
; GCN-DAG: v_mov_b32_e32 [[VS1:v[0-9]+]], s[[#LOAD + 3]]

; GCN-DAG: v_fma_f32 [[RESULT0:v[0-9]+]], s[[#LOAD + 2]], [[VS1]], [[VK0]]
; GCN-DAG: v_mov_b32_e32 [[VK1:v[0-9]+]], 0x45800000
; GCN-DAG: v_fma_f32 [[RESULT1:v[0-9]+]], s[[#LOAD + 2]], [[VS1]], [[VK1]]

; GCN: buffer_store_dword [[RESULT0]]
; GCN: buffer_store_dword [[RESULT1]]
define amdgpu_kernel void @test_s0_s1_k_f32(float addrspace(1)* %out, float %a, float %b) #0 {
  %fma0 = call float @llvm.fma.f32(float %a, float %b, float 1024.0) #1
  %fma1 = call float @llvm.fma.f32(float %a, float %b, float 4096.0) #1
  store volatile float %fma0, float addrspace(1)* %out
  store volatile float %fma1, float addrspace(1)* %out
  ret void
}

; FIXME: Immediate in SGPRs just copied to VGPRs
; GCN-LABEL: {{^}}test_s0_s1_k_f64:
; GCN-DAG: s_load_dwordx2 [[SGPR0:s\[[0-9]+:[0-9]+\]]], s{{\[[0-9]+:[0-9]+\]}}, {{0x13|0x4c}}
; GCN-DAG: s_load_dwordx2 s[[[SGPR1_SUB0:[0-9]+]]:[[SGPR1_SUB1:[0-9]+]]], s{{\[[0-9]+:[0-9]+\]}}, {{0x1d|0x74}}
; GCN-DAG: v_mov_b32_e32 v[[VK0_SUB1:[0-9]+]], 0x40900000
; GCN-DAG: v_mov_b32_e32 v[[VZERO:[0-9]+]], 0{{$}}

; GCN-DAG: v_mov_b32_e32 v[[VS1_SUB0:[0-9]+]], s[[SGPR1_SUB0]]
; GCN-DAG: v_mov_b32_e32 v[[VS1_SUB1:[0-9]+]], s[[SGPR1_SUB1]]
; GCN: v_fma_f64 [[RESULT0:v\[[0-9]+:[0-9]+\]]], [[SGPR0]], v[[[VS1_SUB0]]:[[VS1_SUB1]]], v[[[VZERO]]:[[VK0_SUB1]]]

; Same zero component is re-used for half of each immediate.
; GCN: v_mov_b32_e32 v[[VK1_SUB1:[0-9]+]], 0x40b00000
; GCN: v_fma_f64 [[RESULT1:v\[[0-9]+:[0-9]+\]]], [[SGPR0]], v[[[VS1_SUB0]]:[[VS1_SUB1]]], v[[[VZERO]]:[[VK1_SUB1]]]

; GCN: buffer_store_dwordx2 [[RESULT0]]
; GCN: buffer_store_dwordx2 [[RESULT1]]
define amdgpu_kernel void @test_s0_s1_k_f64(double addrspace(1)* %out, [8 x i32], double %a, [8 x i32], double %b) #0 {
  %fma0 = call double @llvm.fma.f64(double %a, double %b, double 1024.0) #1
  %fma1 = call double @llvm.fma.f64(double %a, double %b, double 4096.0) #1
  store volatile double %fma0, double addrspace(1)* %out
  store volatile double %fma1, double addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
