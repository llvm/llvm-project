; REQUIRES: asserts

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -debug-only=gcn-subtarget < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s 2>&1 | FileCheck -check-prefixes=WARNING %s

; CHECK: Post-MI-sched direction (postra-sched-topdown): topdown
define float @postra-sched-topdown(float %input) nounwind #0 {
  %x = fadd float %input, 1.000000e+00
  ret float %x
}

; CHECK: Post-MI-sched direction (postra-sched-bottomup): bottomup
define float @postra-sched-bottomup(float %input) nounwind #1 {
  %x = fsub float %input, 1.000000e+00
  ret float %x
}

; CHECK: Post-MI-sched direction (postra-sched-bidirectional): bidirectional
define float @postra-sched-bidirectional(float %input) nounwind #2 {
  %x = fadd float %input, 1.000000e+00
  ret float %x
}

; CHECK: Post-MI-sched direction (postra-sched-warning): topdown
; WARNING: invalid value for postRA direction attribute
define float @postra-sched-warning(float %input) nounwind #3 {
  %x = fsub float %input, 1.000000e+00
  ret float %x
}

attributes #0 = {"amdgpu-post-ra-direction"="topdown"}
attributes #1 = {"amdgpu-post-ra-direction"="bottomup"}
attributes #2 = {"amdgpu-post-ra-direction"="bidirectional"}
attributes #3 = {"amdgpu-post-ra-direction"="warning"}
