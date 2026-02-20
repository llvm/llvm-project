; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=null %s 2>&1 | FileCheck %s

; This is a negative test: the VGPR physical register range below requires
; a 13-register (416-bit) VGPR, which doesn't exist.

; CHECK: error: couldn't allocate output register for constraint '{v[0:12]}'

target triple = "amdgcn-amd-amdhsa"

define amdgpu_kernel void @k() {
entry:
  %x = call i416 asm sideeffect "", "={v[0:12]}"()
  ret void
}
