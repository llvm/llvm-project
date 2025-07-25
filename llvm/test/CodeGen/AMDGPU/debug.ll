; RUN: llc < %s -mtriple=amdgcn -mcpu=verde -mattr=dumpcode -filetype=obj | FileCheck --check-prefix=SI %s
; RUN: llc < %s -mtriple=amdgcn -mcpu=tonga -mattr=dumpcode -filetype=obj | FileCheck --check-prefix=SI %s

; Test for a crash in the custom assembly dump code.

; SI: test:
; SI: BB0_0:
; SI: s_endpgm
define amdgpu_kernel void @test(ptr addrspace(1) %out) {
  store i32 0, ptr addrspace(1) %out
  ret void
}
