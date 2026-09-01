; RUN: llc -mtriple=amdgpu9.06 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: llc -mtriple=amdgpu9.08 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s

; REQUIRES: asserts

; OFF: sramecc setting for subtarget: Off

define void @sramecc-subtarget-feature-disabled() #0 {
  ret void
}

attributes #0 = { "target-features"="-sramecc" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.sramecc", i32 0}
