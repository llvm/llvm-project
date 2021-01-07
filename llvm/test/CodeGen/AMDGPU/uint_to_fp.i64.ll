; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck --check-prefixes=GCN,FUNC %s

; FIXME: This should be merged with uint_to_fp.ll, but s_uint_to_fp_v2i64 crashes on r600

; FUNC-LABEL: {{^}}s_uint_to_fp_i64_to_f16:
define amdgpu_kernel void @s_uint_to_fp_i64_to_f16(half addrspace(1)* %out, i64 %in) #0 {
  %result = uitofp i64 %in to half
  store half %result, half addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_uint_to_fp_i64_to_f16:
; GCN: {{buffer|flat}}_load_dwordx2

; GCN: v_ffbh_u32
; GCN: v_ffbh_u32
; GCN: v_cndmask
; GCN: v_cndmask

; GCN-DAG: v_cmp_eq_u64
; GCN-DAG: v_cmp_gt_u64

; GCN: v_add_{{[iu]}}32_e32 [[VR:v[0-9]+]]
; GCN: v_cvt_f16_f32_e32 [[VR_F16:v[0-9]+]], [[VR]]
; GCN: {{buffer|flat}}_store_short {{.*}}[[VR_F16]]
define amdgpu_kernel void @v_uint_to_fp_i64_to_f16(half addrspace(1)* %out, i64 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %in.gep
  %result = uitofp i64 %val to half
  store half %result, half addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}s_uint_to_fp_i64_to_f32:
define amdgpu_kernel void @s_uint_to_fp_i64_to_f32(float addrspace(1)* %out, i64 %in) #0 {
  %result = uitofp i64 %in to float
  store float %result, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_uint_to_fp_i64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2

; GCN: v_ffbh_u32
; GCN: v_ffbh_u32
; GCN: v_cndmask
; GCN: v_cndmask

; GCN-DAG: v_cmp_eq_u64
; GCN-DAG: v_cmp_gt_u64

; GCN: v_add_{{[iu]}}32_e32 [[VR:v[0-9]+]]
; GCN: {{buffer|flat}}_store_dword {{.*}}[[VR]]
define amdgpu_kernel void @v_uint_to_fp_i64_to_f32(float addrspace(1)* %out, i64 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %in.gep
  %result = uitofp i64 %val to float
  store float %result, float addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}s_uint_to_fp_v2i64_to_v2f32:
define amdgpu_kernel void @s_uint_to_fp_v2i64_to_v2f32(<2 x float> addrspace(1)* %out, <2 x i64> %in) #0{
  %result = uitofp <2 x i64> %in to <2 x float>
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_uint_to_fp_v4i64_to_v4f32:
define amdgpu_kernel void @v_uint_to_fp_v4i64_to_v4f32(<4 x float> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr <4 x float>, <4 x float> addrspace(1)* %out, i32 %tid
  %value = load <4 x i64>, <4 x i64> addrspace(1)* %in.gep
  %result = uitofp <4 x i64> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}s_uint_to_fp_v2i64_to_v2f16:
define amdgpu_kernel void @s_uint_to_fp_v2i64_to_v2f16(<2 x half> addrspace(1)* %out, <2 x i64> %in) #0{
  %result = uitofp <2 x i64> %in to <2 x half>
  store <2 x half> %result, <2 x half> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_uint_to_fp_v4i64_to_v4f16:
define amdgpu_kernel void @v_uint_to_fp_v4i64_to_v4f16(<4 x half> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr <4 x half>, <4 x half> addrspace(1)* %out, i32 %tid
  %value = load <4 x i64>, <4 x i64> addrspace(1)* %in.gep
  %result = uitofp <4 x i64> %value to <4 x half>
  store <4 x half> %result, <4 x half> addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
