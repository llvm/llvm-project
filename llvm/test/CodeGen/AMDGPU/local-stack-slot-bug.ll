; RUN: llc -march=amdgcn -mcpu=verde -mattr=+vgpr-spilling -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -mattr=+vgpr-spilling -verify-machineinstrs < %s | FileCheck %s

; This used to fail due to a v_add_i32 instruction with an illegal immediate
; operand that was created during Local Stack Slot Allocation. Test case derived
; from https://bugs.freedesktop.org/show_bug.cgi?id=96602
;
; CHECK-LABEL: {{^}}main:

; CHECK-DAG: v_mov_b32_e32 [[K:v[0-9]+]], 0x200
; CHECK-DAG: v_mov_b32_e32 [[ZERO:v[0-9]+]], 0x400{{$}}
; CHECK-DAG: v_lshlrev_b32_e32 [[BYTES:v[0-9]+]], 2, v0
; CHECK-DAG: v_and_b32_e32 [[CLAMP_IDX:v[0-9]+]], 0x1fc, [[BYTES]]

; CHECK-DAG: v_or_b32_e32 [[LO_OFF:v[0-9]+]], [[CLAMP_IDX]], [[K]]
; CHECK-DAG: v_or_b32_e32 [[HI_OFF:v[0-9]+]], [[CLAMP_IDX]], [[ZERO]]

; CHECK: buffer_load_dword {{v[0-9]+}}, [[LO_OFF]], {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen
; CHECK: buffer_load_dword {{v[0-9]+}}, [[HI_OFF]], {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen
define amdgpu_ps float @main(i32 %idx) {
main_body:
  %v1 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0xBFEA477C60000000, float 0xBFEBE5DC60000000, float 0xBFEC71C720000000, float 0xBFEBE5DC60000000, float 0xBFEA477C60000000, float 0xBFE7A693C0000000, float 0xBFE41CFEA0000000, float 0x3FDF9B13E0000000, float 0x3FDF9B1380000000, float 0x3FD5C53B80000000, float 0x3FD5C53B00000000, float 0x3FC6326AC0000000, float 0x3FC63269E0000000, float 0xBEE05CEB00000000, float 0xBEE086A320000000, float 0xBFC63269E0000000, float 0xBFC6326AC0000000, float 0xBFD5C53B80000000, float 0xBFD5C53B80000000, float 0xBFDF9B13E0000000, float 0xBFDF9B1460000000, float 0xBFE41CFE80000000, float 0x3FE7A693C0000000, float 0x3FEA477C20000000, float 0x3FEBE5DC40000000, float 0x3FEC71C6E0000000, float 0x3FEBE5DC40000000, float 0x3FEA477C20000000, float 0x3FE7A693C0000000, float 0xBFE41CFE80000000>, i32 %idx
  %v2 = extractelement <81 x float> <float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float undef, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFEA0000000, float 0xBFE7A693C0000000, float 0x3FE7A693C0000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFEBE5DC40000000, float 0x3FEBE5DC40000000, float 0xBFEC71C720000000, float 0x3FEC71C6E0000000, float 0xBFEBE5DC60000000, float 0x3FEBE5DC40000000, float 0xBFEA477C20000000, float 0x3FEA477C20000000, float 0xBFE7A693C0000000, float 0x3FE7A69380000000, float 0xBFE41CFEA0000000, float 0xBFDF9B13E0000000, float 0xBFD5C53B80000000, float 0xBFC6326AC0000000, float 0x3EE0789320000000, float 0x3FC6326AC0000000, float 0x3FD5C53B80000000, float 0x3FDF9B13E0000000, float 0x3FE41CFE80000000>, i32 %idx
  %r = fadd float %v1, %v2
  ret float %r
}
