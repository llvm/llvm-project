; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; Test using an integer literal constant.
; Generated ASM should be:
; ADD_INT KC0[2].Z literal.x, 5
; or
; ADD_INT literal.x KC0[2].Z, 5

; CHECK: {{^}}i32_literal:
; CHECK: ADD_INT {{\** *}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, literal.x
; CHECK-NEXT: LSHR
; CHECK-NEXT: 5
define void @i32_literal(i32 addrspace(1)* %out, i32 %in) {
entry:
  %0 = add i32 5, %in
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; Test using a float literal constant.
; Generated ASM should be:
; ADD KC0[2].Z literal.x, 5.0
; or
; ADD literal.x KC0[2].Z, 5.0

; CHECK: {{^}}float_literal:
; CHECK: ADD {{\** *}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, literal.x
; CHECK-NEXT: LSHR
; CHECK-NEXT: 1084227584(5.0
define void @float_literal(float addrspace(1)* %out, float %in) {
entry:
  %0 = fadd float 5.0, %in
  store float %0, float addrspace(1)* %out
  ret void
}

; Make sure inline literals are folded into REG_SEQUENCE instructions.
; CHECK: {{^}}inline_literal_reg_sequence:
; CHECK: MOV {{\** *}}T[[GPR:[0-9]]].X, 0.0
; CHECK-NEXT: MOV {{\** *}}T[[GPR]].Y, 0.0
; CHECK-NEXT: MOV {{\** *}}T[[GPR]].Z, 0.0
; CHECK-NEXT: MOV {{\** *}}T[[GPR]].W, 0.0

define void @inline_literal_reg_sequence(<4 x i32> addrspace(1)* %out) {
entry:
  store <4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32> addrspace(1)* %out
  ret void
}

; CHECK: {{^}}inline_literal_dot4:
; CHECK: DOT4 T[[GPR:[0-9]]].X, 1.0
; CHECK-NEXT: DOT4 T[[GPR]].Y (MASKED), 1.0
; CHECK-NEXT: DOT4 T[[GPR]].Z (MASKED), 1.0
; CHECK-NEXT: DOT4 * T[[GPR]].W (MASKED), 1.0
define void @inline_literal_dot4(float addrspace(1)* %out) {
entry:
  %0 = call float @llvm.AMDGPU.dp4(<4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>)
  store float %0, float addrspace(1)* %out
  ret void
}

declare float @llvm.AMDGPU.dp4(<4 x float>, <4 x float>) #1

attributes #1 = { readnone }
