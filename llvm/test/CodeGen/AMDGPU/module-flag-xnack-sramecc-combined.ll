; Test that xnack and sramecc target ID come from module flags
;
; RUN: llc -mtriple=amdgpu9.0a-amd-amdhsa < %s | FileCheck %s

; Verify the target ID uses module flags (xnack+:sramecc-)
; CHECK: .amdgcn_target "amdgpu9.0a-amd-amdhsa-unknown-gfx90a:sramecc-:xnack+"
; CHECK: amdhsa.target: 'amdgpu9.0a-amd-amdhsa-unknown-gfx90a:sramecc-:xnack+'

define void @foo() {
  ret void
}

!llvm.module.flags = !{!0, !1}
!0 = !{i32 1, !"amdgpu.xnack", i32 1}
!1 = !{i32 1, !"amdgpu.sramecc", i32 0}
