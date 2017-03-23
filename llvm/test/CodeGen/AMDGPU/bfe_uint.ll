; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: {{^}}bfe_def:
; CHECK: BFE_UINT
define amdgpu_kernel void @bfe_def(i32 addrspace(1)* %out, i32 %x) {
entry:
  %0 = lshr i32 %x, 5
  %1 = and i32 %0, 15 ; 0xf
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; This program could be implemented using a BFE_UINT instruction, however
; since the lshr constant + number of bits in the mask is >= 32, it can also be
; implmented with a LSHR instruction, which is better, because LSHR has less
; operands and requires less constants.

; CHECK: {{^}}bfe_shift:
; CHECK-NOT: BFE_UINT
define amdgpu_kernel void @bfe_shift(i32 addrspace(1)* %out, i32 %x) {
entry:
  %0 = lshr i32 %x, 16
  %1 = and i32 %0, 65535 ; 0xffff
  store i32 %1, i32 addrspace(1)* %out
  ret void
}
