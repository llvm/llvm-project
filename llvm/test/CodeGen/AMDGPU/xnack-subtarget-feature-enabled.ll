; RUN: llc -mtriple=amdgpu8.01 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu9.00 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu9.06 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgpu10.10 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s

; REQUIRES: asserts

; ON: xnack setting for subtarget: On
define void @xnack-subtarget-feature-enabled() #0 {
  ret void
}

attributes #0 = { "target-features"="+xnack" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.xnack", i32 1}
