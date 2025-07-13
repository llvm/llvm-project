; REQUIRES: asserts

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -debug-only=machine-scheduler < %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s 2>&1 | FileCheck -check-prefixes=WARNING %s

; CHECK-LABEL: {{^}}postra-sched-topdown:
; CHECK: Post-MI-sched direction (postra-sched-topdown): topdown
define float @postra-sched-topdown(float %input) nounwind #0 {
  %x = fadd float %input, 1.000000e+00
  ret float %x
}

; CHECK-LABEL: {{^}}postra-sched-bottomup:
; CHECK: Post-MI-sched direction (postra-sched-bottomup): bottomup
define float @postra-sched-bottomup(float %input) nounwind #1 {
  %x = fsub float %input, 1.000000e+00
  ret float %x
}

; CHECK-LABEL: {{^}}postra-sched-bidirectional:
; CHECK: Post-MI-sched direction (postra-sched-bidirectional): bidirectional
define float @postra-sched-bidirectional(float %input) nounwind #2 {
  %x = fadd float %input, 1.000000e+00
  ret float %x
}

; CHECK-LABEL: {{^}}postra-sched-warning:
; CHECK: Post-MI-sched direction (postra-sched-warning): default
; WARNING: invalid value for postRa direction attribute
define float @postra-sched-warning(float %input) nounwind #3 {
  %x = fsub float %input, 1.000000e+00
  ret float %x
}

attributes #0 = { alwaysinline nounwind memory(readwrite) "amdgpu-post-ra-direction"="topdown"}
attributes #1 = { alwaysinline nounwind memory(readwrite) "amdgpu-post-ra-direction"="bottomup"}
attributes #2 = { alwaysinline nounwind memory(readwrite) "amdgpu-post-ra-direction"="bidirectional"}
attributes #3 = { alwaysinline nounwind memory(readwrite) "amdgpu-post-ra-direction"="warning"}
