; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_umul24_i32:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, KC0[2].W
; SI: v_mul_u32_u24
define void @test_umul24_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = lshr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = lshr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umul24_i16_sext:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]}}.[[MUL_CHAN:[XYZW]]]
; The result must be sign-extended
; EG: BFE_INT {{[* ]*}}T{{[0-9]}}.{{[XYZW]}}, PV.[[MUL_CHAN]], 0.0, literal.x
; EG: 16

; SI: v_mul_u32_u24_e{{(32|64)}} [[MUL:v[0-9]]], {{[sv][0-9], [sv][0-9]}}
; SI: v_bfe_i32 v{{[0-9]}}, [[MUL]], 0, 16
define void @test_umul24_i16_sext(i32 addrspace(1)* %out, i16 %a, i16 %b) {
entry:
  %mul = mul i16 %a, %b
  %ext = sext i16 %mul to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umul24_i16:
; SI: s_and_b32
; SI: v_mul_u32_u24_e32
; SI: v_and_b32_e32
define void @test_umul24_i16(i32 addrspace(1)* %out, i16 %a, i16 %b) {
entry:
  %mul = mul i16 %a, %b
  %ext = zext i16 %mul to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umul24_i8:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]}}.[[MUL_CHAN:[XYZW]]]
; The result must be sign-extended
; EG: BFE_INT {{[* ]*}}T{{[0-9]}}.{{[XYZW]}}, PV.[[MUL_CHAN]], 0.0, literal.x
; SI: v_mul_u32_u24_e{{(32|64)}} [[MUL:v[0-9]]], {{[sv][0-9], [sv][0-9]}}
; SI: v_bfe_i32 v{{[0-9]}}, [[MUL]], 0, 8

define void @test_umul24_i8(i32 addrspace(1)* %out, i8 %a, i8 %b) {
entry:
  %mul = mul i8 %a, %b
  %ext = sext i8 %mul to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umulhi24_i32_i64:
; SI-NOT: and
; SI: v_mul_hi_u32_u24_e32 [[RESULT:v[0-9]+]],
; SI-NEXT: buffer_store_dword [[RESULT]]

; EG: MULHI_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, KC0[2].W
define void @test_umulhi24_i32_i64(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %a.24 = and i32 %a, 16777215
  %b.24 = and i32 %b, 16777215
  %a.24.i64 = zext i32 %a.24 to i64
  %b.24.i64 = zext i32 %b.24 to i64
  %mul48 = mul i64 %a.24.i64, %b.24.i64
  %mul48.hi = lshr i64 %mul48, 32
  %mul24hi = trunc i64 %mul48.hi to i32
  store i32 %mul24hi, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umulhi24:
; SI-NOT: and
; SI: v_mul_hi_u32_u24_e32 [[RESULT:v[0-9]+]],
; SI-NEXT: buffer_store_dword [[RESULT]]

; EG: MULHI_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].W, KC0[3].Y
define void @test_umulhi24(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %a.24 = and i64 %a, 16777215
  %b.24 = and i64 %b, 16777215
  %mul48 = mul i64 %a.24, %b.24
  %mul48.hi = lshr i64 %mul48, 32
  %mul24.hi = trunc i64 %mul48.hi to i32
  store i32 %mul24.hi, i32 addrspace(1)* %out
  ret void
}

; Multiply with 24-bit inputs and 64-bit output
; FUNC-LABEL: {{^}}test_umul24_i64:
; EG; MUL_UINT24
; EG: MULHI

; SI-NOT: and
; SI-NOT: lshr

; SI-DAG: v_mul_u32_u24_e32
; SI-DAG: v_mul_hi_u32_u24_e32

; SI: buffer_store_dwordx2
define void @test_umul24_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %tmp0 = shl i64 %a, 40
  %a_24 = lshr i64 %tmp0, 40
  %tmp1 = shl i64 %b, 40
  %b_24 = lshr i64 %tmp1, 40
  %tmp2 = mul i64 %a_24, %b_24
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}

; FIXME: Should be able to eliminate the and
; FUNC-LABEL: {{^}}test_umul24_i64_square:
; SI: s_load_dword [[A:s[0-9]+]]
; SI: s_and_b32 [[TRUNC:s[0-9]+]], [[A]], 0xffffff{{$}}
; SI-DAG: v_mul_hi_u32_u24_e64 v{{[0-9]+}}, [[TRUNC]], [[TRUNC]]
; SI-DAG: v_mul_u32_u24_e64 v{{[0-9]+}}, [[TRUNC]], [[TRUNC]]
define void @test_umul24_i64_square(i64 addrspace(1)* %out, i64 %a) {
entry:
  %tmp0 = shl i64 %a, 40
  %a.24 = lshr i64 %tmp0, 40
  %tmp2 = mul i64 %a.24, %a.24
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umulhi16_i32:
; SI: s_and_b32
; SI: s_and_b32
; SI: v_mul_u32_u24_e32 [[MUL24:v[0-9]+]]
; SI: v_lshrrev_b32_e32 v{{[0-9]+}}, 16, [[MUL24]]
define void @test_umulhi16_i32(i16 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %a.16 = and i32 %a, 65535
  %b.16 = and i32 %b, 65535
  %mul = mul i32 %a.16, %b.16
  %hi = lshr i32 %mul, 16
  %mulhi = trunc i32 %hi to i16
  store i16 %mulhi, i16 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umul24_i33:
; SI: s_load_dword s
; SI: s_load_dword s

; SI-NOT: and
; SI-NOT: lshr

; SI-DAG: v_mul_u32_u24_e32 v[[MUL_LO:[0-9]+]],
; SI-DAG: v_mul_hi_u32_u24_e32 v[[MUL_HI:[0-9]+]],
; SI-DAG: v_and_b32_e32 v[[HI:[0-9]+]], 1, v[[MUL_HI]]
; SI: buffer_store_dwordx2 v{{\[}}[[MUL_LO]]:[[HI]]{{\]}}
define void @test_umul24_i33(i64 addrspace(1)* %out, i33 %a, i33 %b) {
entry:
  %tmp0 = shl i33 %a, 9
  %a_24 = lshr i33 %tmp0, 9
  %tmp1 = shl i33 %b, 9
  %b_24 = lshr i33 %tmp1, 9
  %tmp2 = mul i33 %a_24, %b_24
  %ext = zext i33 %tmp2 to i64
  store i64 %ext, i64 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umulhi24_i33:
; SI: s_load_dword s
; SI: s_load_dword s

; SI-NOT: and
; SI-NOT: lshr

; SI: v_mul_hi_u32_u24_e32 v[[MUL_HI:[0-9]+]],
; SI-NEXT: v_and_b32_e32 v[[HI:[0-9]+]], 1, v[[MUL_HI]]
; SI-NEXT: buffer_store_dword v[[HI]]
define void @test_umulhi24_i33(i32 addrspace(1)* %out, i33 %a, i33 %b) {
entry:
  %tmp0 = shl i33 %a, 9
  %a_24 = lshr i33 %tmp0, 9
  %tmp1 = shl i33 %b, 9
  %b_24 = lshr i33 %tmp1, 9
  %tmp2 = mul i33 %a_24, %b_24
  %hi = lshr i33 %tmp2, 32
  %trunc = trunc i33 %hi to i32
  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}
