; RUN: llc -mtriple=amdgcn -mcpu=gfx801 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx900 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx906 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s
; RUN: llc -mtriple=amdgcn -mcpu=gfx1010 -debug-only=gcn-subtarget -o - %s 2>&1 | FileCheck --check-prefix=OFF %s

; REQUIRES: asserts

; OFF: xnack setting for subtarget: Off

define void @xnack-subtarget-feature-disabled() #0 {
  ret void
}

attributes #0 = { "target-features"="-xnack" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu.xnack", i32 0}
