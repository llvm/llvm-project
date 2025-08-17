; RUN: llc -mtriple=amdgcn < %s | FileCheck -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn -mcpu=tonga < %s | FileCheck -check-prefix=SI %s

; SILowerI1Copies was not handling IMPLICIT_DEF
; SI-LABEL: {{^}}br_poison:
; SI: %bb.0:
; SI-NEXT: s_cbranch_scc1
define amdgpu_kernel void @br_poison(ptr addrspace(1) %out, i32 %arg) #0 {
bb:
  br i1 poison, label %bb1, label %bb2

bb1:
  store volatile i32 123, ptr addrspace(1) %out
  ret void

bb2:
  ret void
}

; SI-LABEL: {{^}}br_freeze_poison:
; SI: %bb.0:
; SI-NEXT: s_cbranch_scc1
define amdgpu_kernel void @br_freeze_poison(ptr addrspace(1) %out, i32 %arg) #0 {
bb:
  %undef = freeze i1 poison
  br i1 %undef, label %bb1, label %bb2

bb1:
  store volatile i32 123, ptr addrspace(1) %out
  ret void

bb2:
  ret void
}

attributes #0 = { nounwind }
