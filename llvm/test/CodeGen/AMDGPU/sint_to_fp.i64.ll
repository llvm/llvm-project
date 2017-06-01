; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s

; FIXME: This should be merged with sint_to_fp.ll, but s_sint_to_fp_v2i64 crashes on r600

; FUNC-LABEL: {{^}}s_sint_to_fp_i64_to_f16:
define void @s_sint_to_fp_i64_to_f16(half addrspace(1)* %out, i64 %in) #0 {
  %result = sitofp i64 %in to half
  store half %result, half addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_sint_to_fp_i64_to_f16:
; GCN: {{buffer|flat}}_load_dwordx2

; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
; GCN: v_xor_b32

; GCN: v_ffbh_u32
; GCN: v_ffbh_u32
; GCN: v_cndmask
; GCN: v_cndmask

; GCN-DAG: v_cmp_eq_u64
; GCN-DAG: v_cmp_lt_u64

; GCN: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, v{{[0-9]+}}
; GCN: v_cndmask_b32_e{{32|64}} [[SIGN_SEL:v[0-9]+]],
; GCN: v_cvt_f16_f32_e32 [[SIGN_SEL_F16:v[0-9]+]], [[SIGN_SEL]]
; GCN: {{buffer|flat}}_store_short {{.*}}[[SIGN_SEL_F16]]
define void @v_sint_to_fp_i64_to_f16(half addrspace(1)* %out, i64 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr half, half addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %in.gep
  %result = sitofp i64 %val to half
  store half %result, half addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}s_sint_to_fp_i64_to_f32:
define void @s_sint_to_fp_i64_to_f32(float addrspace(1)* %out, i64 %in) #0 {
  %result = sitofp i64 %in to float
  store float %result, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_sint_to_fp_i64_to_f32:
; GCN: {{buffer|flat}}_load_dwordx2

; GCN: v_ashrrev_i32_e32 v{{[0-9]+}}, 31, v{{[0-9]+}}
; GCN: v_xor_b32

; GCN: v_ffbh_u32
; GCN: v_ffbh_u32
; GCN: v_cndmask
; GCN: v_cndmask

; GCN-DAG: v_cmp_eq_u64
; GCN-DAG: v_cmp_lt_u64

; GCN: v_xor_b32_e32 v{{[0-9]+}}, 0x80000000, v{{[0-9]+}}
; GCN: v_cndmask_b32_e{{32|64}} [[SIGN_SEL:v[0-9]+]],
; GCN: {{buffer|flat}}_store_dword {{.*}}[[SIGN_SEL]]
define void @v_sint_to_fp_i64_to_f32(float addrspace(1)* %out, i64 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %in.gep
  %result = sitofp i64 %val to float
  store float %result, float addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}s_sint_to_fp_v2i64_to_v2f32:
; GCN-NOT: v_and_b32_e32 v{{[0-9]+}}, -1,
define void @s_sint_to_fp_v2i64_to_v2f32(<2 x float> addrspace(1)* %out, <2 x i64> %in) #0{
  %result = sitofp <2 x i64> %in to <2 x float>
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_sint_to_fp_v4i64_to_v4f32:
define void @v_sint_to_fp_v4i64_to_v4f32(<4 x float> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr <4 x float>, <4 x float> addrspace(1)* %out, i32 %tid
  %value = load <4 x i64>, <4 x i64> addrspace(1)* %in.gep
  %result = sitofp <4 x i64> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}s_sint_to_fp_v2i64_to_v2f16:
; GCN-NOT: v_and_b32_e32 v{{[0-9]+}}, -1,
define void @s_sint_to_fp_v2i64_to_v2f16(<2 x half> addrspace(1)* %out, <2 x i64> %in) #0{
  %result = sitofp <2 x i64> %in to <2 x half>
  store <2 x half> %result, <2 x half> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_sint_to_fp_v4i64_to_v4f16:
define void @v_sint_to_fp_v4i64_to_v4f16(<4 x half> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr <4 x half>, <4 x half> addrspace(1)* %out, i32 %tid
  %value = load <4 x i64>, <4 x i64> addrspace(1)* %in.gep
  %result = sitofp <4 x i64> %value to <4 x half>
  store <4 x half> %result, <4 x half> addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
