; RUN: llc < %s -mtriple=amdgpu6.01 -mattr=dumpcode -filetype=obj | FileCheck --check-prefix=SI %s
; RUN: llc < %s -mtriple=amdgpu8.02 -mattr=dumpcode -filetype=obj | FileCheck --check-prefix=SI %s

; Test for a crash in the custom assembly dump code.

; SI: test:
; SI: BB0_0:
; SI: s_endpgm
define amdgpu_kernel void @test(ptr addrspace(1) %out) {
  store i32 0, ptr addrspace(1) %out
  ret void
}
