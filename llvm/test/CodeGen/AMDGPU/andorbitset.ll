; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}s_clear_msb:
; SI: s_bitset0_b32 s{{[0-9]+}}, 31
define amdgpu_kernel void @s_clear_msb(i32 addrspace(1)* %out, i32 %in) {
  %x = and i32 %in, 2147483647
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_set_msb:
; SI: s_bitset1_b32 s{{[0-9]+}}, 31
define amdgpu_kernel void @s_set_msb(i32 addrspace(1)* %out, i32 %in) {
  %x = or i32 %in, 2147483648
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_clear_lsb:
; SI: s_and_b32 s{{[0-9]+}}, s{{[0-9]+}}, -2
define amdgpu_kernel void @s_clear_lsb(i32 addrspace(1)* %out, i32 %in) {
  %x = and i32 %in, 4294967294
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_set_lsb:
; SI: s_or_b32 s{{[0-9]+}}, s{{[0-9]+}}, 1
define amdgpu_kernel void @s_set_lsb(i32 addrspace(1)* %out, i32 %in) {
  %x = or i32 %in, 1
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_clear_midbit:
; SI: s_bitset0_b32 s{{[0-9]+}}, 8
define amdgpu_kernel void @s_clear_midbit(i32 addrspace(1)* %out, i32 %in) {
  %x = and i32 %in, 4294967039
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_set_midbit:
; SI: s_bitset1_b32 s{{[0-9]+}}, 8
define amdgpu_kernel void @s_set_midbit(i32 addrspace(1)* %out, i32 %in) {
  %x = or i32 %in, 256
  store i32 %x, i32 addrspace(1)* %out
  ret void
}

@gv = external addrspace(1) global i32

; Make sure there's no verifier error with an undef source.
; SI-LABEL: {{^}}bitset_verifier_error:
; SI-NOT:   %bb.1:
; SI:       s_bitset0_b32 s{{[0-9]+}}, 31
define void @bitset_verifier_error() local_unnamed_addr #0 {
bb:
  %i = call float @llvm.fabs.f32(float undef) #0
  %i1 = bitcast float %i to i32
  store i32 %i1, i32 addrspace(1)* @gv
  br label %bb2

bb2:
  %i3 = call float @llvm.fabs.f32(float undef) #0
  %i4 = fcmp fast ult float %i3, 0x3FEFF7CEE0000000
  br i1 %i4, label %bb5, label %bb6

bb5:
  unreachable

bb6:
  unreachable
}

declare float @llvm.fabs.f32(float) #0

attributes #0 = { nounwind readnone speculatable willreturn }
