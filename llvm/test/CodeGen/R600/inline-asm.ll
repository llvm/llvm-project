; RUN: llc < %s -march=r600 -mcpu=SI -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK: {{^}}inline_asm:
; CHECK: s_endpgm
; CHECK: s_endpgm
define void @inline_asm(i32 addrspace(1)* %out) {
entry:
  store i32 5, i32 addrspace(1)* %out
  call void asm sideeffect "s_endpgm", ""()
  ret void
}
