; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck %s --check-prefix=SI --check-prefix=FUNC
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s --check-prefix=SI --check-prefix=FUNC
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck %s --check-prefix=EG --check-prefix=FUNC

declare float @llvm.fabs.f32(float) #0

; FUNC-LABEL: {{^}}fp_to_sint_i32:
; EG: FLT_TO_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; SI: v_cvt_i32_f32_e32
; SI: s_endpgm
define void @fp_to_sint_i32(i32 addrspace(1)* %out, float %in) {
  %conv = fptosi float %in to i32
  store i32 %conv, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_sint_i32_fabs:
; SI: v_cvt_i32_f32_e64 v{{[0-9]+}}, |s{{[0-9]+}}|{{$}}
define void @fp_to_sint_i32_fabs(i32 addrspace(1)* %out, float %in) {
  %in.fabs = call float @llvm.fabs.f32(float %in) #0
  %conv = fptosi float %in.fabs to i32
  store i32 %conv, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_sint_v2i32:
; EG: FLT_TO_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; SI: v_cvt_i32_f32_e32
; SI: v_cvt_i32_f32_e32
define void @fp_to_sint_v2i32(<2 x i32> addrspace(1)* %out, <2 x float> %in) {
  %result = fptosi <2 x float> %in to <2 x i32>
  store <2 x i32> %result, <2 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_sint_v4i32:
; EG: FLT_TO_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_INT {{\** *}}T{{[0-9]+\.[XYZW]}}
; EG: FLT_TO_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; EG: FLT_TO_INT {{\** *}}T{{[0-9]+\.[XYZW], PV\.[XYZW]}}
; SI: v_cvt_i32_f32_e32
; SI: v_cvt_i32_f32_e32
; SI: v_cvt_i32_f32_e32
; SI: v_cvt_i32_f32_e32
define void @fp_to_sint_v4i32(<4 x i32> addrspace(1)* %out, <4 x float> addrspace(1)* %in) {
  %value = load <4 x float>, <4 x float> addrspace(1) * %in
  %result = fptosi <4 x float> %value to <4 x i32>
  store <4 x i32> %result, <4 x i32> addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}fp_to_sint_i64:

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

; Check that the compiler doesn't crash with a "cannot select" error
; SI: s_endpgm
define void @fp_to_sint_i64 (i64 addrspace(1)* %out, float %in) {
entry:
  %0 = fptosi float %in to i64
  store i64 %0, i64 addrspace(1)* %out
  ret void
}

; FUNC: {{^}}fp_to_sint_v2i64:
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

; SI: s_endpgm
define void @fp_to_sint_v2i64(<2 x i64> addrspace(1)* %out, <2 x float> %x) {
  %conv = fptosi <2 x float> %x to <2 x i64>
  store <2 x i64> %conv, <2 x i64> addrspace(1)* %out
  ret void
}

; FUNC: {{^}}fp_to_sint_v4i64:
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

; SI: s_endpgm
define void @fp_to_sint_v4i64(<4 x i64> addrspace(1)* %out, <4 x float> %x) {
  %conv = fptosi <4 x float> %x to <4 x i64>
  store <4 x i64> %conv, <4 x i64> addrspace(1)* %out
  ret void
}

attributes #0 = { nounwind readnone }
