; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -verify-machineinstrs < %s | FileCheck --check-prefix=GCN %s

; GCN-LABEL: {{^}}scalar_and_or_not_i16
; GCN: s_not_b32
; GCN-NEXT: s_lshr_b32
; GCN-NEXT: s_and_b32
; GCN-NEXT: s_andn2_b32
define amdgpu_kernel void @scalar_and_or_not_i16(ptr addrspace(1) %out, i16 %x, i16 %y, i16 %z) {
entry:
  %not_z = xor i16 %z, -1
  %or_y_not_z = or i16 %y, %not_z
  %and_result = and i16 %x, %or_y_not_z
  store i16 %and_result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}scalar_and_or_not_i32
; GCN: s_andn2_b32
; GCN-NEXT: s_andn2_b32
define amdgpu_kernel void @scalar_and_or_not_i32(ptr addrspace(1) %out, i32 %x, i32 %y, i32 %z) {
entry:
  %not_z = xor i32 %z, -1
  %or_y_not_z = or i32 %y, %not_z
  %and_result = and i32 %x, %or_y_not_z
  store i32 %and_result, ptr addrspace(1) %out, align 4
  ret void
}

; GCN-LABEL: {{^}}scalar_and_or_not_i64
; GCN: s_andn2_b64
; GCN-NEXT: s_andn2_b64
define amdgpu_kernel void @scalar_and_or_not_i64(ptr addrspace(1) %out, i64 %x, i64 %y, i64 %z) {
entry:
  %not_z = xor i64 %z, -1
  %or_y_not_z = or i64 %y, %not_z
  %and_result = and i64 %x, %or_y_not_z
  store i64 %and_result, ptr addrspace(1) %out, align 4
  ret void
}
