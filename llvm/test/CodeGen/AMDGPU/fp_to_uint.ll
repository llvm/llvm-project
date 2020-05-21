; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefixes=GCN,FUNC,SI
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefixes=GCN,FUNC,VI
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -allow-deprecated-dag-overlap %s -check-prefix=EG -check-prefix=FUNC

declare float @llvm.fabs.f32(float) #1

; FUNC-LABEL: {{^}}fp_to_uint_f32_to_i32:
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}

; GCN: v_cvt_u32_f32_e32
; GCN: s_endpgm
define amdgpu_kernel void @fp_to_uint_f32_to_i32 (i32 addrspace(1)* %out, float %in) {
  %conv = fptoui float %in to i32
  store i32 %conv, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_uint_v2f32_to_v2i32:
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

; GCN: v_cvt_u32_f32_e32
; GCN: v_cvt_u32_f32_e32
define amdgpu_kernel void @fp_to_uint_v2f32_to_v2i32(<2 x i32> addrspace(1)* %out, <2 x float> %in) {
  %result = fptoui <2 x float> %in to <2 x i32>
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_uint_v4f32_to_v4i32:
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_UINT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; GCN: v_cvt_u32_f32_e32
; GCN: v_cvt_u32_f32_e32
; GCN: v_cvt_u32_f32_e32
; GCN: v_cvt_u32_f32_e32

define amdgpu_kernel void @fp_to_uint_v4f32_to_v4i32(<4 x i32> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %value = load <4 x float>, <4 x float> addrspace(1) * %in
  %result = fptoui <4 x float> %value to <4 x i32>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC: {{^}}fp_to_uint_f32_to_i64:
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; GCN: s_endpgm
define amdgpu_kernel void @fp_to_uint_f32_to_i64(i64 addrspace(1)* %out, float %x) {
  %conv = fptoui float %x to i64
  store i64 %conv, i64 addrspace(1)* %out
  ret void
}

; FUNC: {{^}}fp_to_uint_v2f32_to_v2i64:
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; GCN: s_endpgm
define amdgpu_kernel void @fp_to_uint_v2f32_to_v2i64(<2 x i64> addrspace(1)* %out, <2 x float> %x) {
  %conv = fptoui <2 x float> %x to <2 x i64>
  store <2 x i64> %conv, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC: {{^}}fp_to_uint_v4f32_to_v4i64:
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT
; EG-DAG: AND_INT
; EG-DAG: LSHR
; EG-DAG: SUB_INT
; EG-DAG: AND_INT
; EG-DAG: ASHR
; EG-DAG: AND_INT
; EG-DAG: OR_INT
; EG-DAG: SUB_INT
; EG-DAG: LSHL
; EG-DAG: LSHL
; EG-DAG: SUB_INT
; EG-DAG: LSHR
; EG-DAG: LSHR
; EG-DAG: SETGT_UINT
; EG-DAG: SETGT_INT
; EG-DAG: XOR_INT
; EG-DAG: XOR_INT
; EG-DAG: SUB_INT
; EG-DAG: SUB_INT
; EG-DAG: CNDE_INT
; EG-DAG: CNDE_INT

; GCN: s_endpgm
define amdgpu_kernel void @fp_to_uint_v4f32_to_v4i64(<4 x i64> addrspace(1)* %out, <4 x float> %x) {
  %conv = fptoui <4 x float> %x to <4 x i64>
  store <4 x i64> %conv, <4 x i64> addrspace(1)* %out
  ret void
}


; FUNC-LABEL: {{^}}fp_to_uint_f32_to_i1:
; GCN: v_cmp_eq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, 1.0, s{{[0-9]+}}

; EG: AND_INT
; EG: SETE_DX10 {{[*]?}} T{{[0-9]+}}.{{[XYZW]}}, KC0[2].Z, 1.0,
define amdgpu_kernel void @fp_to_uint_f32_to_i1(i1 addrspace(1)* %out, float %in) #0 {
  %conv = fptoui float %in to i1
  store i1 %conv, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_uint_fabs_f32_to_i1:
; GCN: v_cmp_eq_f32_e64 s{{\[[0-9]+:[0-9]+\]}}, 1.0, |s{{[0-9]+}}|
define amdgpu_kernel void @fp_to_uint_fabs_f32_to_i1(i1 addrspace(1)* %out, float %in) #0 {
  %in.fabs = call float @llvm.fabs.f32(float %in)
  %conv = fptoui float %in.fabs to i1
  store i1 %conv, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_uint_f32_to_i16:
; COM: The reason different instructions are used on SI and VI is because for
; COM: SI fp_to_uint is legalized by the type legalizer and for VI it is
; COM: legalized by the dag legalizer and they legalize fp_to_uint differently.
; SI: v_cvt_u32_f32_e32 [[VAL:v[0-9]+]], s{{[0-9]+}}
; VI: v_cvt_i32_f32_e32 [[VAL:v[0-9]+]], s{{[0-9]+}}
; GCN: buffer_store_short [[VAL]]
define amdgpu_kernel void @fp_to_uint_f32_to_i16(i16 addrspace(1)* %out, float %in) #0 {
  %uint = fptoui float %in to i16
  store i16 %uint, i16 addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
