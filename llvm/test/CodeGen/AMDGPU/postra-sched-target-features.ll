; REQUIRES: asserts

; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 -debug-only=gcn-subtarget < %s 2>&1 | FileCheck %s

; CHECK: Post-MI-sched direction: topdown
define float @postra-sched-topdown(float %input) nounwind #0 {
  %x = fadd float %input, 1.000000e+00
  ret float %x
}

; CHECK: Post-MI-sched direction: bottomup
define float @postra-sched-bottomup(float %input) nounwind #1 {
  %x = fsub float %input, 1.000000e+00
  ret float %x
}

; CHECK: Post-MI-sched direction: bidirectional
define float @postra-sched-bidirectional(float %input) nounwind #2 {
  %x = fadd float %input, 1.000000e+00
  ret float %x
}

attributes #0 = { "target-features"="+postra-top-down" }
attributes #1 = { "target-features"="+postra-bottom-up" }
attributes #2 = { "target-features"="+postra-bidirectional" }
