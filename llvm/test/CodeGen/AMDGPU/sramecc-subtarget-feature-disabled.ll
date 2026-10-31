; RUN: llc -mtriple=amdgpu9.06 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: llc -mtriple=amdgpu9.08 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s

; SRAMECC is hardwired on for gfx12.5; the module flag is ignored.
; RUN: llc -mtriple=amdgpu12.50 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s

; REQUIRES: asserts

; OFF: sramecc setting for subtarget: Off
; ON: sramecc setting for subtarget: On

define void @sramecc-subtarget-feature-disabled() #0 {
  ret void
}

attributes #0 = { "target-features"="-sramecc" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.sramecc", i32 0}
