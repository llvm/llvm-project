; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI -check-prefix=FUNC %s

; FIXME: This should be merged with uint_to_fp.ll, but s_uint_to_fp_v2i64 crashes on r600

; FUNC-LABEL: {{^}}s_uint_to_fp_i64_to_f32:
define void @s_uint_to_fp_i64_to_f32(float addrspace(1)* %out, i64 %in) #0 {
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

; GCN-DAG: v_cmp_eq_i64
; GCN-DAG: v_cmp_lt_u64

; GCN: v_add_i32_e32 [[VR:v[0-9]+]]
; GCN: {{buffer|flat}}_store_dword [[VR]]
define void @v_uint_to_fp_i64_to_f32(float addrspace(1)* %out, i64 addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr float, float addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %in.gep
  %result = uitofp i64 %val to float
  store float %result, float addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}s_uint_to_fp_v2i64:
define void @s_uint_to_fp_v2i64(<2 x float> addrspace(1)* %out, <2 x i64> %in) #0{
  %result = uitofp <2 x i64> %in to <2 x float>
  store <2 x float> %result, <2 x float> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_uint_to_fp_v4i64:
define void @v_uint_to_fp_v4i64(<4 x float> addrspace(1)* %out, <4 x i64> addrspace(1)* %in) #0 {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %in.gep = getelementptr <4 x i64>, <4 x i64> addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr <4 x float>, <4 x float> addrspace(1)* %out, i32 %tid
  %value = load <4 x i64>, <4 x i64> addrspace(1)* %in.gep
  %result = uitofp <4 x i64> %value to <4 x float>
  store <4 x float> %result, <4 x float> addrspace(1)* %out.gep
  ret void
}

declare i32 @llvm.r600.read.tidig.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
