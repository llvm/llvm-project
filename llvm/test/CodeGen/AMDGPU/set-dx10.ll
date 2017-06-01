; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; These tests check that floating point comparisons which are used by select
; to store integer true (-1) and false (0) values are lowered to one of the
; SET*DX10 instructions.

; CHECK: {{^}}fcmp_une_select_fptosi:
; CHECK: LSHR
; CHECK-NEXT: SETNE_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, literal.y,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_une_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp une float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_une_select_i32:
; CHECK: LSHR
; CHECK-NEXT: SETNE_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, literal.y,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_une_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp une float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_oeq_select_fptosi:
; CHECK: LSHR
; CHECK-NEXT: SETE_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, literal.y,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_oeq_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp oeq float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_oeq_select_i32:
; CHECK: LSHR
; CHECK-NEXT: SETE_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, literal.y,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_oeq_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp oeq float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_ogt_select_fptosi:
; CHECK: LSHR
; CHECK-NEXT: SETGT_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, literal.y,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ogt_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ogt float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_ogt_select_i32:
; CHECK: LSHR
; CHECK-NEXT: SETGT_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, literal.y,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ogt_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ogt float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_oge_select_fptosi:
; CHECK: LSHR
; CHECK-NEXT: SETGE_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, literal.y,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_oge_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp oge float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_oge_select_i32:
; CHECK: LSHR
; CHECK-NEXT: SETGE_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, KC0[2].Z, literal.y,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_oge_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp oge float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_ole_select_fptosi:
; CHECK: LSHR
; CHECK-NEXT: SETGE_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, literal.y, KC0[2].Z,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ole_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ole float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_ole_select_i32:
; CHECK: LSHR
; CHECK-NEXT: SETGE_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, literal.y, KC0[2].Z,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_ole_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp ole float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_olt_select_fptosi:
; CHECK: LSHR
; CHECK-NEXT: SETGT_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, literal.y, KC0[2].Z,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_olt_select_fptosi(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp olt float %in, 5.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  store i32 %3, i32 addrspace(1)* %out
  ret void
}

; CHECK: {{^}}fcmp_olt_select_i32:
; CHECK: LSHR
; CHECK-NEXT: SETGT_DX10 * {{\** *}}T{{[0-9]+\.[XYZW]}}, literal.y, KC0[2].Z,
; CHECK-NEXT: 1084227584(5.000000e+00)
define void @fcmp_olt_select_i32(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp olt float %in, 5.0
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}
