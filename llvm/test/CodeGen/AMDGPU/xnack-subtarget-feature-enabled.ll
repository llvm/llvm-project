; RUN: llc -mtriple=amdgcn -mcpu=gfx801 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=ON %s

; REQUIRES: asserts

; ON: xnack setting for subtarget: On
define void @xnack-subtarget-feature-enabled() #0 {
  ret void
}

attributes #0 = { "target-features"="+xnack" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.xnack", i32 1}
