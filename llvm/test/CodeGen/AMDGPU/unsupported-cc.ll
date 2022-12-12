; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; These tests are for condition codes that are not supported by the hardware

; CHECK-LABEL: {{^}}slt:
; CHECK: LSHR
; CHECK-NEXT: SETGT_INT {{\** *}}T{{[0-9]+\.[XYZW]}}, {{literal\.[xy]}}, KC0[2].Z
; CHECK-NEXT: 5(7.006492e-45)
define amdgpu_kernel void @slt(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = icmp slt i32 %in, 5
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}ult_i32:
; CHECK: LSHR
; CHECK-NEXT: SETGT_UINT {{\** *}}T{{[0-9]+\.[XYZW]}}, {{literal\.[xy]}}, KC0[2].Z
; CHECK-NEXT: 5(7.006492e-45)
define amdgpu_kernel void @ult_i32(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = icmp ult i32 %in, 5
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}ult_float:
; CHECK: SETGE * T{{[0-9]}}.[[CHAN:[XYZW]]], KC0[2].Z, literal.x
; CHECK-NEXT: 1084227584(5.000000e+00)
; CHECK-NEXT: SETE T{{[0-9]\.[XYZW]}}, PV.[[CHAN]], 0.0
; CHECK-NEXT: LSHR *
define amdgpu_kernel void @ult_float(ptr addrspace(1) %out, float %in) {
entry:
  %0 = fcmp ult float %in, 5.0
  %1 = select i1 %0, float 1.0, float 0.0
  store float %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}ult_float_native:
; CHECK: LSHR
; CHECK-NEXT: SETGE {{\*? *}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, {{literal\.[xy]}}
; CHECK-NEXT: 1084227584(5.000000e+00)
define amdgpu_kernel void @ult_float_native(ptr addrspace(1) %out, float %in) {
entry:
  %0 = fcmp ult float %in, 5.0
  %1 = select i1 %0, float 0.0, float 1.0
  store float %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}olt:
; CHECK: LSHR
; CHECK-NEXT: SETGT {{\*? *}}T{{[0-9]+\.[XYZW]}}, {{literal\.[xy]}}, KC0[2].Z
; CHECK-NEXT: 1084227584(5.000000e+00)
define amdgpu_kernel void @olt(ptr addrspace(1) %out, float %in) {
entry:
  %0 = fcmp olt float %in, 5.0
  %1 = select i1 %0, float 1.0, float 0.0
  store float %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}sle:
; CHECK: LSHR
; CHECK-NEXT: SETGT_INT {{\** *}}T{{[0-9]+\.[XYZW]}}, {{literal\.[xy]}}, KC0[2].Z
; CHECK-NEXT: 6(8.407791e-45)
define amdgpu_kernel void @sle(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = icmp sle i32 %in, 5
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}ule_i32:
; CHECK: LSHR
; CHECK-NEXT: SETGT_UINT {{\** *}}T{{[0-9]+\.[XYZW]}}, {{literal\.[xy]}}, KC0[2].Z
; CHECK-NEXT: 6(8.407791e-45)
define amdgpu_kernel void @ule_i32(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = icmp ule i32 %in, 5
  %1 = select i1 %0, i32 -1, i32 0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}ule_float:
; CHECK: SETGT * T{{[0-9]}}.[[CHAN:[XYZW]]], KC0[2].Z, literal.x
; CHECK-NEXT: 1084227584(5.000000e+00)
; CHECK-NEXT: SETE T{{[0-9]\.[XYZW]}}, PV.[[CHAN]], 0.0
; CHECK-NEXT: LSHR *
define amdgpu_kernel void @ule_float(ptr addrspace(1) %out, float %in) {
entry:
  %0 = fcmp ule float %in, 5.0
  %1 = select i1 %0, float 1.0, float 0.0
  store float %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}ule_float_native:
; CHECK: LSHR
; CHECK-NEXT: SETGT {{\*? *}}T{{[0-9]\.[XYZW]}}, KC0[2].Z, {{literal\.[xy]}}
; CHECK-NEXT: 1084227584(5.000000e+00)
define amdgpu_kernel void @ule_float_native(ptr addrspace(1) %out, float %in) {
entry:
  %0 = fcmp ule float %in, 5.0
  %1 = select i1 %0, float 0.0, float 1.0
  store float %1, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}ole:
; CHECK: LSHR
; CHECK-NEXT: SETGE {{\*? *}}T{{[0-9]\.[XYZW]}}, {{literal\.[xy]}}, KC0[2].Z
; CHECK-NEXT:1084227584(5.000000e+00)
define amdgpu_kernel void @ole(ptr addrspace(1) %out, float %in) {
entry:
  %0 = fcmp ole float %in, 5.0
  %1 = select i1 %0, float 1.0, float 0.0
  store float %1, ptr addrspace(1) %out
  ret void
}
