; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=VI -check-prefix=FUNC -check-prefix=GCN %s
; RUN: llc -march=r600 -mcpu=cypress -verify-machineinstrs < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

declare i8 @llvm.ctlz.i8(i8, i1) nounwind readnone

declare i32 @llvm.ctlz.i32(i32, i1) nounwind readnone
declare <2 x i32> @llvm.ctlz.v2i32(<2 x i32>, i1) nounwind readnone
declare <4 x i32> @llvm.ctlz.v4i32(<4 x i32>, i1) nounwind readnone

declare i64 @llvm.ctlz.i64(i64, i1) nounwind readnone
declare <2 x i64> @llvm.ctlz.v2i64(<2 x i64>, i1) nounwind readnone
declare <4 x i64> @llvm.ctlz.v4i64(<4 x i64>, i1) nounwind readnone

declare i32 @llvm.r600.read.tidig.x() nounwind readnone

; FUNC-LABEL: {{^}}s_ctlz_zero_undef_i32:
; GCN: s_load_dword [[VAL:s[0-9]+]],
; GCN: s_flbit_i32_b32 [[SRESULT:s[0-9]+]], [[VAL]]
; GCN: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[SRESULT]]
; GCN: buffer_store_dword [[VRESULT]],
; GCN: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @s_ctlz_zero_undef_i32(i32 addrspace(1)* noalias %out, i32 %val) nounwind {
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32:
; GCN: buffer_load_dword [[VAL:v[0-9]+]],
; GCN: v_ffbh_u32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[RESULT]],
; GCN: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+\.[XYZW]]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @v_ctlz_zero_undef_i32(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr, align 4
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  store i32 %ctlz, i32 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_v2i32:
; GCN: buffer_load_dwordx2
; GCN: v_ffbh_u32_e32
; GCN: v_ffbh_u32_e32
; GCN: buffer_store_dwordx2
; GCN: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+]]{{\.[XYZW]}}
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @v_ctlz_zero_undef_v2i32(<2 x i32> addrspace(1)* noalias %out, <2 x i32> addrspace(1)* noalias %valptr) nounwind {
  %val = load <2 x i32>, <2 x i32> addrspace(1)* %valptr, align 8
  %ctlz = call <2 x i32> @llvm.ctlz.v2i32(<2 x i32> %val, i1 true) nounwind readnone
  store <2 x i32> %ctlz, <2 x i32> addrspace(1)* %out, align 8
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_v4i32:
; GCN: buffer_load_dwordx4
; GCN: v_ffbh_u32_e32
; GCN: v_ffbh_u32_e32
; GCN: v_ffbh_u32_e32
; GCN: v_ffbh_u32_e32
; GCN: buffer_store_dwordx4
; GCN: s_endpgm
; EG: MEM_RAT_CACHELESS STORE_RAW [[RESULT:T[0-9]+]]{{\.[XYZW]}}
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
; EG: FFBH_UINT {{\*? *}}[[RESULT]]
define amdgpu_kernel void @v_ctlz_zero_undef_v4i32(<4 x i32> addrspace(1)* noalias %out, <4 x i32> addrspace(1)* noalias %valptr) nounwind {
  %val = load <4 x i32>, <4 x i32> addrspace(1)* %valptr, align 16
  %ctlz = call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %val, i1 true) nounwind readnone
  store <4 x i32> %ctlz, <4 x i32> addrspace(1)* %out, align 16
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i8:
; GCN: buffer_load_ubyte [[VAL:v[0-9]+]],
; GCN: v_ffbh_u32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GCN: buffer_store_byte [[RESULT]],
define amdgpu_kernel void @v_ctlz_zero_undef_i8(i8 addrspace(1)* noalias %out, i8 addrspace(1)* noalias %valptr) nounwind {
  %val = load i8, i8 addrspace(1)* %valptr
  %ctlz = call i8 @llvm.ctlz.i8(i8 %val, i1 true) nounwind readnone
  store i8 %ctlz, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_ctlz_zero_undef_i64:
; GCN: s_load_dwordx2 s{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}, s{{\[[0-9]+:[0-9]+\]}}, {{0xb|0x2c}}
; GCN-DAG: v_cmp_eq_u32_e64 vcc, s[[HI]], 0{{$}}
; GCN-DAG: s_flbit_i32_b32 [[FFBH_LO:s[0-9]+]], s[[LO]]
; GCN-DAG: s_add_i32 [[ADD:s[0-9]+]], [[FFBH_LO]], 32
; GCN-DAG: s_flbit_i32_b32 [[FFBH_HI:s[0-9]+]], s[[HI]]
; GCN-DAG: v_mov_b32_e32 [[VFFBH_LO:v[0-9]+]], [[FFBH_LO]]
; GCN-DAG: v_mov_b32_e32 [[VFFBH_HI:v[0-9]+]], [[FFBH_HI]]
; GCN-DAG: v_cndmask_b32_e32 v[[CTLZ:[0-9]+]], [[VFFBH_HI]], [[VFFBH_LO]]
; GCN-DAG: v_mov_b32_e32 v[[CTLZ_HI:[0-9]+]], 0{{$}}
; GCN: {{buffer|flat}}_store_dwordx2 v{{\[}}[[CTLZ]]:[[CTLZ_HI]]{{\]}}
define amdgpu_kernel void @s_ctlz_zero_undef_i64(i64 addrspace(1)* noalias %out, i64 %val) nounwind {
  %ctlz = call i64 @llvm.ctlz.i64(i64 %val, i1 true)
  store i64 %ctlz, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}s_ctlz_zero_undef_i64_trunc:
define amdgpu_kernel void @s_ctlz_zero_undef_i64_trunc(i32 addrspace(1)* noalias %out, i64 %val) nounwind {
  %ctlz = call i64 @llvm.ctlz.i64(i64 %val, i1 true)
  %trunc = trunc i64 %ctlz to i32
  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i64:
; GCN-DAG: {{buffer|flat}}_load_dwordx2 v{{\[}}[[LO:[0-9]+]]:[[HI:[0-9]+]]{{\]}}
; GCN-DAG: v_cmp_eq_u32_e64 [[CMPHI:s\[[0-9]+:[0-9]+\]]], 0, v[[HI]]
; GCN-DAG: v_ffbh_u32_e32 [[FFBH_LO:v[0-9]+]], v[[LO]]
; GCN-DAG: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, 32, [[FFBH_LO]]
; GCN-DAG: v_ffbh_u32_e32 [[FFBH_HI:v[0-9]+]], v[[HI]]
; GCN-DAG: v_cndmask_b32_e64 v[[CTLZ:[0-9]+]], [[FFBH_HI]], [[FFBH_LO]]
; GCN: {{buffer|flat}}_store_dwordx2 {{.*}}v{{\[}}[[CTLZ]]:[[CTLZ_HI:[0-9]+]]{{\]}}
define amdgpu_kernel void @v_ctlz_zero_undef_i64(i64 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr i64, i64 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %in.gep
  %ctlz = call i64 @llvm.ctlz.i64(i64 %val, i1 true)
  store i64 %ctlz, i64 addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i64_trunc:
define amdgpu_kernel void @v_ctlz_zero_undef_i64_trunc(i32 addrspace(1)* noalias %out, i64 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %in.gep = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %out.gep = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %val = load i64, i64 addrspace(1)* %in.gep
  %ctlz = call i64 @llvm.ctlz.i64(i64 %val, i1 true)
  %trunc = trunc i64 %ctlz to i32
  store i32 %trunc, i32 addrspace(1)* %out.gep
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32_sel_eq_neg1:
; GCN: buffer_load_dword [[VAL:v[0-9]+]],
; GCN: v_ffbh_u32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[RESULT]],
 define amdgpu_kernel void @v_ctlz_zero_undef_i32_sel_eq_neg1(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  %cmp = icmp eq i32 %val, 0
  %sel = select i1 %cmp, i32 -1, i32 %ctlz
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32_sel_ne_neg1:
; GCN: buffer_load_dword [[VAL:v[0-9]+]],
; GCN: v_ffbh_u32_e32 [[RESULT:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[RESULT]],
define amdgpu_kernel void @v_ctlz_zero_undef_i32_sel_ne_neg1(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  %cmp = icmp ne i32 %val, 0
  %sel = select i1 %cmp, i32 %ctlz, i32 -1
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i8_sel_eq_neg1:
; GCN: {{buffer|flat}}_load_ubyte [[VAL:v[0-9]+]],
; GCN: v_ffbh_u32_e32 [[FFBH:v[0-9]+]], [[VAL]]
; GCN: {{buffer|flat}}_store_byte [[FFBH]],
define amdgpu_kernel void @v_ctlz_zero_undef_i8_sel_eq_neg1(i8 addrspace(1)* noalias %out, i8 addrspace(1)* noalias %valptr) nounwind {
  %tid = call i32 @llvm.r600.read.tidig.x()
  %valptr.gep = getelementptr i8, i8 addrspace(1)* %valptr, i32 %tid
  %val = load i8, i8 addrspace(1)* %valptr.gep
  %ctlz = call i8 @llvm.ctlz.i8(i8 %val, i1 true) nounwind readnone
  %cmp = icmp eq i8 %val, 0
  %sel = select i1 %cmp, i8 -1, i8 %ctlz
  store i8 %sel, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32_sel_eq_neg1_two_use:
; GCN: buffer_load_dword [[VAL:v[0-9]+]],
; GCN-DAG: v_ffbh_u32_e32 [[RESULT0:v[0-9]+]], [[VAL]]
; GCN-DAG: v_cmp_eq_u32_e32 vcc, 0, [[VAL]]
; GCN-DAG: v_cndmask_b32_e64 [[RESULT1:v[0-9]+]], 0, 1, vcc
; GCN-DAG: buffer_store_dword [[RESULT0]]
; GCN-DAG: buffer_store_byte [[RESULT1]]
; GCN: s_endpgm
 define amdgpu_kernel void @v_ctlz_zero_undef_i32_sel_eq_neg1_two_use(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  %cmp = icmp eq i32 %val, 0
  %sel = select i1 %cmp, i32 -1, i32 %ctlz
  store volatile i32 %sel, i32 addrspace(1)* %out
  store volatile i1 %cmp, i1 addrspace(1)* undef
  ret void
}

; Selected on wrong constant
; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32_sel_eq_0:
; GCN: buffer_load_dword
; GCN: v_ffbh_u32_e32
; GCN: v_cmp
; GCN: v_cndmask
; GCN: buffer_store_dword
 define amdgpu_kernel void @v_ctlz_zero_undef_i32_sel_eq_0(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  %cmp = icmp eq i32 %val, 0
  %sel = select i1 %cmp, i32 0, i32 %ctlz
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; Selected on wrong constant
; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32_sel_ne_0:
; GCN: buffer_load_dword
; GCN: v_ffbh_u32_e32
; GCN: v_cmp
; GCN: v_cndmask
; GCN: buffer_store_dword
define amdgpu_kernel void @v_ctlz_zero_undef_i32_sel_ne_0(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  %cmp = icmp ne i32 %val, 0
  %sel = select i1 %cmp, i32 %ctlz, i32 0
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; Compare on wrong constant
; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32_sel_eq_cmp_non0:
; GCN: buffer_load_dword
; GCN: v_ffbh_u32_e32
; GCN: v_cmp
; GCN: v_cndmask
; GCN: buffer_store_dword
 define amdgpu_kernel void @v_ctlz_zero_undef_i32_sel_eq_cmp_non0(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  %cmp = icmp eq i32 %val, 1
  %sel = select i1 %cmp, i32 0, i32 %ctlz
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}

; Selected on wrong constant
; FUNC-LABEL: {{^}}v_ctlz_zero_undef_i32_sel_ne_cmp_non0:
; GCN: buffer_load_dword
; GCN: v_ffbh_u32_e32
; GCN: v_cmp
; GCN: v_cndmask
; GCN: buffer_store_dword
define amdgpu_kernel void @v_ctlz_zero_undef_i32_sel_ne_cmp_non0(i32 addrspace(1)* noalias %out, i32 addrspace(1)* noalias %valptr) nounwind {
  %val = load i32, i32 addrspace(1)* %valptr
  %ctlz = call i32 @llvm.ctlz.i32(i32 %val, i1 true) nounwind readnone
  %cmp = icmp ne i32 %val, 1
  %sel = select i1 %cmp, i32 %ctlz, i32 0
  store i32 %sel, i32 addrspace(1)* %out
  ret void
}
