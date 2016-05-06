; RUN: llc < %s -march=amdgcn -verify-machineinstrs | FileCheck %s --check-prefix=SI --check-prefix=FUNC
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=SI --check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s --check-prefix=EG --check-prefix=FUNC
; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s --check-prefix=EG --check-prefix=FUNC

; FUNC-LABEL: {{^}}u32_mul24:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, KC0[2].W
; SI: v_mul_u32_u24

define void @u32_mul24(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = lshr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = lshr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i16_mul24:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]}}.[[MUL_CHAN:[XYZW]]]
; The result must be sign-extended
; EG: BFE_INT {{[* ]*}}T{{[0-9]}}.{{[XYZW]}}, PV.[[MUL_CHAN]], 0.0, literal.x
; EG: 16
; SI: v_mul_u32_u24_e{{(32|64)}} [[MUL:v[0-9]]], {{[sv][0-9], [sv][0-9]}}
; SI: v_bfe_i32 v{{[0-9]}}, [[MUL]], 0, 16
define void @i16_mul24(i32 addrspace(1)* %out, i16 %a, i16 %b) {
entry:
  %0 = mul i16 %a, %b
  %1 = sext i16 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}i8_mul24:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]}}.[[MUL_CHAN:[XYZW]]]
; The result must be sign-extended
; EG: BFE_INT {{[* ]*}}T{{[0-9]}}.{{[XYZW]}}, PV.[[MUL_CHAN]], 0.0, literal.x
; SI: v_mul_u32_u24_e{{(32|64)}} [[MUL:v[0-9]]], {{[sv][0-9], [sv][0-9]}}
; SI: v_bfe_i32 v{{[0-9]}}, [[MUL]], 0, 8

define void @i8_mul24(i32 addrspace(1)* %out, i8 %a, i8 %b) {
entry:
  %0 = mul i8 %a, %b
  %1 = sext i8 %0 to i32
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; Multiply with 24-bit inputs and 64-bit output
; FUNC_LABEL: {{^}}mul24_i64:
; EG; MUL_UINT24
; EG: MULHI
; FIXME: SI support 24-bit mulhi

; SI-DAG: v_mul_u32_u24
; SI-DAG: v_mul_hi_u32
; SI: s_endpgm
define void @mul24_i64(i64 addrspace(1)* %out, i64 %a, i64 %b, i64 %c) {
entry:
  %tmp0 = shl i64 %a, 40
  %a_24 = lshr i64 %tmp0, 40
  %tmp1 = shl i64 %b, 40
  %b_24 = lshr i64 %tmp1, 40
  %tmp2 = mul i64 %a_24, %b_24
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}
