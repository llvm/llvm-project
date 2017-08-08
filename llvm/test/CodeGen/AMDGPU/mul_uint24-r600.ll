; RUN: llc -march=r600 -mcpu=cayman < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_umul24_i32:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, KC0[2].W
define amdgpu_kernel void @test_umul24_i32(i32 addrspace(1)* %out, i32 %a, i32 %b) {
entry:
  %0 = shl i32 %a, 8
  %a_24 = lshr i32 %0, 8
  %1 = shl i32 %b, 8
  %b_24 = lshr i32 %1, 8
  %2 = mul i32 %a_24, %b_24
  store i32 %2, i32 addrspace(1)* %out
  ret void
}

; The result must be sign-extended.
; FUNC-LABEL: {{^}}test_umul24_i16_sext:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]}}.[[MUL_CHAN:[XYZW]]]
; EG: BFE_INT {{[* ]*}}T{{[0-9]}}.{{[XYZW]}}, PV.[[MUL_CHAN]], 0.0, literal.x
; EG: 16
define amdgpu_kernel void @test_umul24_i16_sext(i32 addrspace(1)* %out, i16 %a, i16 %b) {
entry:
  %mul = mul i16 %a, %b
  %ext = sext i16 %mul to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; The result must be sign-extended.
; FUNC-LABEL: {{^}}test_umul24_i8:
; EG: MUL_UINT24 {{[* ]*}}T{{[0-9]}}.[[MUL_CHAN:[XYZW]]]
; EG: BFE_INT {{[* ]*}}T{{[0-9]}}.{{[XYZW]}}, PV.[[MUL_CHAN]], 0.0, literal.x
define amdgpu_kernel void @test_umul24_i8(i32 addrspace(1)* %out, i8 %a, i8 %b) {
entry:
  %mul = mul i8 %a, %b
  %ext = sext i8 %mul to i32
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}test_umulhi24_i32_i64:
; EG: MULHI_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, KC0[2].W
define amdgpu_kernel void @test_umulhi24_i32_i64(i32 addrspace(1)* %out, i32 %a, i32 %b) {
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
; EG: MULHI_UINT24 {{[* ]*}}T{{[0-9]\.[XYZW]}}, KC0[2].W, KC0[3].Y
define amdgpu_kernel void @test_umulhi24(i32 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %a.24 = and i64 %a, 16777215
  %b.24 = and i64 %b, 16777215
  %mul48 = mul i64 %a.24, %b.24
  %mul48.hi = lshr i64 %mul48, 32
  %mul24.hi = trunc i64 %mul48.hi to i32
  store i32 %mul24.hi, i32 addrspace(1)* %out
  ret void
}

; Multiply with 24-bit inputs and 64-bit output.
; FUNC-LABEL: {{^}}test_umul24_i64:
; EG; MUL_UINT24
; EG: MULHI
define amdgpu_kernel void @test_umul24_i64(i64 addrspace(1)* %out, i64 %a, i64 %b) {
entry:
  %tmp0 = shl i64 %a, 40
  %a_24 = lshr i64 %tmp0, 40
  %tmp1 = shl i64 %b, 40
  %b_24 = lshr i64 %tmp1, 40
  %tmp2 = mul i64 %a_24, %b_24
  store i64 %tmp2, i64 addrspace(1)* %out
  ret void
}
