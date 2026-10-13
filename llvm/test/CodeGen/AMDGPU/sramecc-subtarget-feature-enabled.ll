; RUN: llc -mtriple=amdgpu9.06 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu9.08 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu12.50 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s

; REQUIRES: asserts

; ON: sramecc setting for subtarget: On
define void @sramecc-subtarget-feature-enabled() #0 {
  ret void
}

attributes #0 = { "target-features"="+sramecc" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.sramecc", i32 1}
