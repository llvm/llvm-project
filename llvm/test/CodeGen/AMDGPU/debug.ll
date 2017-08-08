; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs -mattr=dumpcode -filetype=obj | FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs -mattr=dumpcode -filetype=obj | FileCheck --check-prefix=SI --check-prefix=FUNC %s

; Test for a crash in the custom assembly dump code.

; SI: s_endpgm
define amdgpu_kernel void @test(i32 addrspace(1)* %out) {
  store i32 0, i32 addrspace(1)* %out
  ret void
}
