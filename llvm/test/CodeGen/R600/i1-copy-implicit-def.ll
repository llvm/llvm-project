; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SILowerI1Copies was not handling IMPLICIT_DEF
; SI-LABEL: {{^}}br_implicit_def:
; SI: BB#0:
; SI-NEXT: s_and_saveexec_b64
; SI-NEXT: s_xor_b64
; SI-NEXT: BB#1:
define void @br_implicit_def(i32 addrspace(1)* %out, i32 %arg) #0 {
bb:
  br i1 undef, label %bb1, label %bb2

bb1:
  store volatile i32 123, i32 addrspace(1)* %out
  ret void

bb2:
  ret void
}

attributes #0 = { nounwind }
