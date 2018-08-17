; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}test_fmed3:
; GCN: v_med3_f32 v{{[0-9]+}}, s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
define amdgpu_kernel void @test_fmed3(float addrspace(1)* %out, float %src0, float %src1, float %src2) #1 {
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0, float %src1, float %src2)
  store float %med3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fmed3_srcmods:
; GCN: v_med3_f32 v{{[0-9]+}}, -s{{[0-9]+}}, |v{{[0-9]+}}|, -|v{{[0-9]+}}|
define amdgpu_kernel void @test_fmed3_srcmods(float addrspace(1)* %out, float %src0, float %src1, float %src2) #1 {
  %src0.fneg = fsub float -0.0, %src0
  %src1.fabs = call float @llvm.fabs.f32(float %src1)
  %src2.fabs = call float @llvm.fabs.f32(float %src2)
  %src2.fneg.fabs = fsub float -0.0, %src2.fabs
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0.fneg, float %src1.fabs, float %src2.fneg.fabs)
  store float %med3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fneg_fmed3:
; GCN: v_med3_f32 v{{[0-9]+}}, -s{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
define amdgpu_kernel void @test_fneg_fmed3(float addrspace(1)* %out, float %src0, float %src1, float %src2) #1 {
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0, float %src1, float %src2)
  %neg.med3 = fsub float -0.0, %med3
  store float %neg.med3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fneg_fmed3_multi_use:
; GCN: v_med3_f32 [[MED3:v[0-9]+]], -s{{[0-9]+}}, -v{{[0-9]+}}, -v{{[0-9]+}}
; GCN: v_mul_f32_e32 v{{[0-9]+}}, -4.0, [[MED3]]
define amdgpu_kernel void @test_fneg_fmed3_multi_use(float addrspace(1)* %out, float %src0, float %src1, float %src2) #1 {
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0, float %src1, float %src2)
  %neg.med3 = fsub float -0.0, %med3
  %med3.user = fmul float %med3, 4.0
  store volatile float %med3.user, float addrspace(1)* %out
  store volatile float %neg.med3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fabs_fmed3:
; GCN: v_med3_f32 [[MED3:v[0-9]+]], s{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}
; GCN: v_and_b32_e32 v{{[0-9]+}}, 0x7fffffff, [[MED3]]
define amdgpu_kernel void @test_fabs_fmed3(float addrspace(1)* %out, float %src0, float %src1, float %src2) #1 {
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0, float %src1, float %src2)
  %fabs.med3 = call float @llvm.fabs.f32(float %med3)
  store float %fabs.med3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fneg_fmed3_rr_0:
; GCN: v_bfrev_b32_e32 [[NEG0:v[0-9]+]], 1
; GCN: v_med3_f32 v{{[0-9]+}}, -s{{[0-9]+}}, -v{{[0-9]+}}, [[NEG0]]
define amdgpu_kernel void @test_fneg_fmed3_rr_0(float addrspace(1)* %out, float %src0, float %src1) #1 {
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0, float %src1, float 0.0)
  %neg.med3 = fsub float -0.0, %med3
  store float %neg.med3, float addrspace(1)* %out
  ret void
}

; FIXME: Worse off from folding this
; GCN-LABEL: {{^}}test_fneg_fmed3_rr_0_foldable_user:
; GCN: v_bfrev_b32_e32 [[NEG0:v[0-9]+]], 1
; GCN: v_med3_f32 [[MED3:v[0-9]+]], -s{{[0-9]+}}, -v{{[0-9]+}}, [[NEG0]]
; GCN: v_mul_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, [[MED3]]
define amdgpu_kernel void @test_fneg_fmed3_rr_0_foldable_user(float addrspace(1)* %out, float %src0, float %src1, float %mul.arg) #1 {
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0, float %src1, float 0.0)
  %neg.med3 = fsub float -0.0, %med3
  %mul = fmul float %neg.med3, %mul.arg
  store float %mul, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fneg_fmed3_r_inv2pi_0:
; GCN-DAG: v_bfrev_b32_e32 [[NEG0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[NEG_INV:v[0-9]+]], 0xbe22f983
; GCN: v_med3_f32 v{{[0-9]+}}, -s{{[0-9]+}}, [[NEG_INV]], [[NEG0]]
define amdgpu_kernel void @test_fneg_fmed3_r_inv2pi_0(float addrspace(1)* %out, float %src0) #1 {
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0, float 0x3FC45F3060000000, float 0.0)
  %neg.med3 = fsub float -0.0, %med3
  store float %neg.med3, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}test_fneg_fmed3_r_inv2pi_0_foldable_user:
; GCN-DAG: v_bfrev_b32_e32 [[NEG0:v[0-9]+]], 1
; GCN-DAG: v_mov_b32_e32 [[NEG_INV:v[0-9]+]], 0xbe22f983
; GCN: v_med3_f32 [[MED3:v[0-9]+]], -s{{[0-9]+}}, [[NEG_INV]], [[NEG0]]
; GCN: v_mul_f32_e32 v{{[0-9]+}}, s{{[0-9]+}}, [[MED3]]
define amdgpu_kernel void @test_fneg_fmed3_r_inv2pi_0_foldable_user(float addrspace(1)* %out, float %src0, float %mul.arg) #1 {
  %med3 = call float @llvm.amdgcn.fmed3.f32(float %src0, float 0x3FC45F3060000000, float 0.0)
  %neg.med3 = fsub float -0.0, %med3
  %mul = fmul float %neg.med3, %mul.arg
  store float %mul, float addrspace(1)* %out
  ret void
}

declare float @llvm.amdgcn.fmed3.f32(float, float, float) #0
declare float @llvm.fabs.f32(float) #0

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind }
