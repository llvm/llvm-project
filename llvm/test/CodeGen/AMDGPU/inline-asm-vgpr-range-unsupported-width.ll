; RUN: not llc -mtriple=amdgpu9.00-amd-amdhsa -filetype=null %s 2>&1 | FileCheck %s

; This is a negative test: the VGPR physical register range below requires
; a 13-register (416-bit) VGPR, which doesn't exist.

; CHECK: error: could not allocate output register for constraint '{v[0:12]}'

define amdgpu_kernel void @k() {
entry:
  %x = call i416 asm sideeffect "", "={v[0:12]}"()
  ret void
}
