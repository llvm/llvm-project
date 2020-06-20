; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=SI-NOSDWA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=SI -check-prefix=SI-SDWA  -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=EG -check-prefix=FUNC %s

declare i7 @llvm.cttz.i7(i7, i1) nounwind readnone
declare i8 @llvm.cttz.i8(i8, i1) nounwind readnone
declare i16 @llvm.cttz.i16(i16, i1) nounwind readnone
declare i32 @llvm.cttz.i32(i32, i1) nounwind readnone
declare i64 @llvm.cttz.i64(i64, i1) nounwind readnone
declare <2 x i32> @llvm.cttz.v2i32(<2 x i32>, i1) nounwind readnone
declare <4 x i32> @llvm.cttz.v4i32(<4 x i32>, i1) nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; FUNC-LABEL: {{^}}s_cttz_zero_undef_i32:
; SI: s_load_dword [[VAL:s[0-9]+]],
; SI: s_ff1_i32_b32 [[SRESULT:s[0-9]+]], [[VAL]]
; SI: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; SI: buffer_store_dword [[VRESULT]],
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @s_cttz_zero_undef_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %cttz = call i32 @llvm.cttz.i32(i32 %val, i1 true) nounwind readnone
  store i32 %cttz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_i32:
; SI: {{buffer|flat}}_load_dword [[VAL:v[0-9]+]],
; SI: v_ffbl_b32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[RESULT]],
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @v_cttz_zero_undef_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr i32, i32 addrspace(1)* %valptr, i32 %tid
  %val = load i32, i32 addrspace(1)* %in.gep, align 4
  %cttz = call i32 @llvm.cttz.i32(i32 %val, i1 true) nounwind readnone
  store i32 %cttz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_v2i32:
; SI: {{buffer|flat}}_load_dwordx2
; SI: v_ffbl_b32_e32
; SI: v_ffbl_b32_e32
; SI: buffer_store_dwordx2
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+]]{{\.[XYZW]}}
; EG: FFBL_INT {{\*? *}}[[RESULT]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @v_cttz_zero_undef_v2i32(<2 x i32> addrspace(1)* noalias %out, <2 x i32> addrspace(1)* noalias %valptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <2 x i32>, <2 x i32> addrspace(1)* %valptr, i32 %tid
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %in.gep, align 8
  %cttz = call <2 x i32> @llvm.cttz.v2i32(<2 x i32> %val, i1 true) nounwind readnone
  store <2 x i32> %cttz, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_v4i32:
; SI: {{buffer|flat}}_load_dwordx4
; SI: v_ffbl_b32_e32
; SI: v_ffbl_b32_e32
; SI: v_ffbl_b32_e32
; SI: v_ffbl_b32_e32
; SI: buffer_store_dwordx4
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+]]{{\.[XYZW]}}
; EG: FFBL_INT {{\*? *}}[[RESULT]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @v_cttz_zero_undef_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %valptr) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %in.gep = getelementptr <4 x i32>, <4 x i32> addrspace(1)* %valptr, i32 %tid
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %in.gep, align 16
  %cttz = call <4 x i32> @llvm.cttz.v4i32(<4 x i32> %val, i1 true) nounwind readnone
  store <4 x i32> %cttz, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}s_cttz_zero_undef_i8_with_select:
; SI: s_ff1_i32_b32 s{{[0-9]+}}, s{{[0-9]+}}
; EG: MEM_RAT MSKOR
; EG: FFBL_INT
define amdgpu_kernel void @s_cttz_zero_undef_i8_with_select(i8 addrspace(1)* noalias %out, i8 %val) nounwind {
  %cttz = tail call i8 @llvm.cttz.i8(i8 %val, i1 true) nounwind readnone
  %cttz_ret = icmp ne i8 %val, 0
  %ret = select i1 %cttz_ret, i8 %cttz, i8 32
  store i8 %cttz, i8 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_cttz_zero_undef_i16_with_select:
; SI: s_ff1_i32_b32 s{{[0-9]+}}, s{{[0-9]+}}
; EG: MEM_RAT MSKOR
; EG: FFBL_INT
define amdgpu_kernel void @s_cttz_zero_undef_i16_with_select(i16 addrspace(1)* noalias %out, i16 %val) nounwind {
  %cttz = tail call i16 @llvm.cttz.i16(i16 %val, i1 true) nounwind readnone
  %cttz_ret = icmp ne i16 %val, 0
  %ret = select i1 %cttz_ret, i16 %cttz, i16 32
  store i16 %cttz, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_cttz_zero_undef_i32_with_select:
; SI: s_ff1_i32_b32
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBL_INT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @s_cttz_zero_undef_i32_with_select(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %cttz = tail call i32 @llvm.cttz.i32(i32 %val, i1 true) nounwind readnone
  %cttz_ret = icmp ne i32 %val, 0
  %ret = select i1 %cttz_ret, i32 %cttz, i32 32
  store i32 %cttz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}s_cttz_zero_undef_i64_with_select:
; SI: s_ff1_i32_b32 s{{[0-9]+}}, s{{[0-9]+}}
; SI: s_ff1_i32_b32 s{{[0-9]+}}, s{{[0-9]+}}
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
define amdgpu_kernel void @s_cttz_zero_undef_i64_with_select(i64 addrspace(1)* noalias %out, i64 %val) nounwind {
  %cttz = tail call i64 @llvm.cttz.i64(i64 %val, i1 true) nounwind readnone
  %cttz_ret = icmp ne i64 %val, 0
  %ret = select i1 %cttz_ret, i64 %cttz, i64 32
  store i64 %cttz, i64 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_i8_with_select:
; SI-NOSDWA: v_ffbl_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; SI-SDWA: v_ffbl_b32_e32
; EG: MEM_RAT MSKOR
define amdgpu_kernel void @v_cttz_zero_undef_i8_with_select(i8 addrspace(1)* noalias %out, i8 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i8, i8 addrspace(1)* %arrayidx, align 1
  %cttz = tail call i8 @llvm.cttz.i8(i8 %val, i1 true) nounwind readnone
  %cttz_ret = icmp ne i8 %val, 0
  %ret = select i1 %cttz_ret, i8 %cttz, i8 32
  store i8 %ret, i8 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_i16_with_select:
; SI-NOSDWA: v_ffbl_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; SI-SDWA: v_ffbl_b32_e32
; EG: MEM_RAT MSKOR
define amdgpu_kernel void @v_cttz_zero_undef_i16_with_select(i16 addrspace(1)* noalias %out, i16 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i16, i16 addrspace(1)* %arrayidx, align 1
  %cttz = tail call i16 @llvm.cttz.i16(i16 %val, i1 true) nounwind readnone
  %cttz_ret = icmp ne i16 %val, 0
  %ret = select i1 %cttz_ret, i16 %cttz, i16 32
  store i16 %ret, i16 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_i32_with_select:
; SI-DAG: v_ffbl_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; SI-DAG: v_cmp_ne_u32_e32 vcc, 0
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
define amdgpu_kernel void @v_cttz_zero_undef_i32_with_select(i32 addrspace(1)* noalias %out, i32 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i32, i32 addrspace(1)* %arrayidx, align 1
  %cttz = tail call i32 @llvm.cttz.i32(i32 %val, i1 true) nounwind readnone
  %cttz_ret = icmp ne i32 %val, 0
  %ret = select i1 %cttz_ret, i32 %cttz, i32 32
  store i32 %ret, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_zero_undef_i64_with_select:
; SI-NOSDWA: v_or_b32_e32
; SI-NOSDWA: v_or_b32_e32
; SI-NOSDWA: v_or_b32_e32
; SI-NOSDWA: v_or_b32_e32
; SI-NOSDWA: v_or_b32_e32 [[VAL1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; SI-NOSDWA: v_or_b32_e32 [[VAL2:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; SI-NOSDWA: v_ffbl_b32_e32 v{{[0-9]+}}, [[VAL1]]
; SI-NOSDWA: v_ffbl_b32_e32 v{{[0-9]+}}, [[VAL2]]
; SI-SDWA: v_or_b32_e32
; SI-SDWA: v_or_b32_sdwa
; SI-SDWA: v_or_b32_e32 [[VAL1:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; SI-SDWA: v_ffbl_b32_e32 v{{[0-9]+}}, [[VAL1]]
; SI-SDWA: v_or_b32_e32
; SI-SDWA: v_or_b32_sdwa
; SI-SDWA: v_or_b32_e32 [[VAL2:v[0-9]+]], v{{[0-9]+}}, v{{[0-9]+}}
; SI-SDWA: v_ffbl_b32_e32 v{{[0-9]+}}, [[VAL2]]
; SI: v_cmp_eq_u32_e32 vcc, 0
; SI: v_cmp_ne_u64_e32 vcc, 0
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
define amdgpu_kernel void @v_cttz_zero_undef_i64_with_select(i64 addrspace(1)* noalias %out, i64 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i64, i64 addrspace(1)* %arrayidx, align 1
  %cttz = tail call i64 @llvm.cttz.i64(i64 %val, i1 true) nounwind readnone
  %cttz_ret = icmp ne i64 %val, 0
  %ret = select i1 %cttz_ret, i64 %cttz, i64 32
  store i64 %ret, i64 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_i32_sel_eq_neg1:
; SI: v_ffbl_b32_e32 v{{[0-9]+}}, [[VAL:v[0-9]+]]
; SI: v_cmp_ne_u32_e32 vcc, 0, [[VAL]]
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW
; EG: FFBL_INT
define amdgpu_kernel void @v_cttz_i32_sel_eq_neg1(i32 addrspace(1)* noalias %out, i32 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i32, i32 addrspace(1)* %arrayidx, align 1
  %ctlz = call i32 @llvm.cttz.i32(i32 %val, i1 false) nounwind readnone
  %cmp = icmp eq i32 %val, 0
  %sel = select i1 %cmp, i32 -1, i32 %ctlz
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_i32_sel_ne_neg1:
; SI: v_ffbl_b32_e32 v{{[0-9]+}}, [[VAL:v[0-9]+]]
; SI: v_cmp_ne_u32_e32 vcc, 0, [[VAL]]
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW
; EG: FFBL_INT
define amdgpu_kernel void @v_cttz_i32_sel_ne_neg1(i32 addrspace(1)* noalias %out, i32 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i32, i32 addrspace(1)* %arrayidx, align 1
  %ctlz = call i32 @llvm.cttz.i32(i32 %val, i1 false) nounwind readnone
  %cmp = icmp ne i32 %val, 0
  %sel = select i1 %cmp, i32 %ctlz, i32 -1
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_i32_sel_ne_bitwidth:
; SI: v_ffbl_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; SI: v_cmp
; SI: v_cndmask
; SI: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW
; EG: FFBL_INT
define amdgpu_kernel void @v_cttz_i32_sel_ne_bitwidth(i32 addrspace(1)* noalias %out, i32 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i32, i32 addrspace(1)* %arrayidx, align 1
  %ctlz = call i32 @llvm.cttz.i32(i32 %val, i1 false) nounwind readnone
  %cmp = icmp ne i32 %ctlz, 32
  %sel = select i1 %cmp, i32 %ctlz, i32 -1
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_i8_sel_eq_neg1:
; SI: {{buffer|flat}}_load_ubyte
; SI-NOSDWA: v_ffbl_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; SI-SDWA: v_ffbl_b32_e32
; EG: MEM_RAT MSKOR
; EG: FFBL_INT
 define amdgpu_kernel void @v_cttz_i8_sel_eq_neg1(i8 addrspace(1)* noalias %out, i8 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i8, i8 addrspace(1)* %arrayidx, align 1
  %ctlz = call i8 @llvm.cttz.i8(i8 %val, i1 false) nounwind readnone
  %cmp = icmp eq i8 %val, 0
  %sel = select i1 %cmp, i8 -1, i8 %ctlz
  store i8 %sel, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_cttz_i16_sel_eq_neg1:
; SI: {{buffer|flat}}_load_ubyte
; SI: v_ffbl_b32_e32 v{{[0-9]+}}, v{{[0-9]+}}
; SI: buffer_store_short
; EG: MEM_RAT MSKOR
; EG: FFBL_INT
 define amdgpu_kernel void @v_cttz_i16_sel_eq_neg1(i16 addrspace(1)* noalias %out, i16 addrspace(1)* nocapture readonly %arrayidx) nounwind {
  %val = load i16, i16 addrspace(1)* %arrayidx, align 1
  %ctlz = call i16 @llvm.cttz.i16(i16 %val, i1 false) nounwind readnone
  %cmp = icmp eq i16 %val, 0
  %sel = select i1 %cmp, i16 -1, i16 %ctlz
  store i16 %sel, i16 addrspace(1)* %out
  ret void
}


