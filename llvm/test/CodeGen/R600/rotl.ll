; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck --check-prefix=R600 -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}rotl_i32:
; R600: SUB_INT {{\** T[0-9]+\.[XYZW]}}, literal.x
; R600-NEXT: 32
; R600: BIT_ALIGN_INT {{T[0-9]+\.[XYZW]}}, KC0[2].Z, KC0[2].Z, PV.{{[XYZW]}}

; SI: s_sub_i32 [[SDST:s[0-9]+]], 32, {{[s][0-9]+}}
; SI: v_mov_b32_e32 [[VDST:v[0-9]+]], [[SDST]]
; SI: v_alignbit_b32 {{v[0-9]+, [s][0-9]+, s[0-9]+}}, [[VDST]]
define void @rotl_i32(i32 addrspace(1)* %in, i32 %x, i32 %y) {
entry:
  %0 = shl i32 %x, %y
  %1 = sub i32 32, %y
  %2 = lshr i32 %x, %1
  %3 = or i32 %0, %2
  store i32 %3, i32 addrspace(1)* %in
  ret void
}

; FUNC-LABEL: {{^}}rotl_v2i32:
; SI-DAG: s_sub_i32
; SI-DAG: s_sub_i32
; SI-DAG: v_alignbit_b32
; SI-DAG: v_alignbit_b32
; SI: s_endpgm
define void @rotl_v2i32(<2 x i32> addrspace(1)* %in, <2 x i32> %x, <2 x i32> %y) {
entry:
  %0 = shl <2 x i32> %x, %y
  %1 = sub <2 x i32> <i32 32, i32 32>, %y
  %2 = lshr <2 x i32> %x, %1
  %3 = or <2 x i32> %0, %2
  store <2 x i32> %3, <2 x i32> addrspace(1)* %in
  ret void
}

; FUNC-LABEL: {{^}}rotl_v4i32:
; SI-DAG: s_sub_i32
; SI-DAG: v_alignbit_b32
; SI-DAG: s_sub_i32
; SI-DAG: v_alignbit_b32
; SI-DAG: s_sub_i32
; SI-DAG: v_alignbit_b32
; SI-DAG: s_sub_i32
; SI-DAG: v_alignbit_b32
; SI: s_endpgm
define void @rotl_v4i32(<4 x i32> addrspace(1)* %in, <4 x i32> %x, <4 x i32> %y) {
entry:
  %0 = shl <4 x i32> %x, %y
  %1 = sub <4 x i32> <i32 32, i32 32, i32 32, i32 32>, %y
  %2 = lshr <4 x i32> %x, %1
  %3 = or <4 x i32> %0, %2
  store <4 x i32> %3, <4 x i32> addrspace(1)* %in
  ret void
}
