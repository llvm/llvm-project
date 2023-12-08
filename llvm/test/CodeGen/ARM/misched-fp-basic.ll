; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm-eabi -mcpu=cortex-a9 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > \
; RUN:   /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK_A9
; RUN: llc < %s -mtriple=arm-eabi -mcpu=swift -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > \
; RUN:   /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK_SWIFT
; RUN: llc < %s -mtriple=arm-eabi -mcpu=cortex-r52 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > \
; RUN:   /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK_R52
;
; Check the latency of instructions for processors with sched-models
;
; Function Attrs: norecurse nounwind readnone
define i32 @foo(float %a, float %b, float %c, i32 %d) local_unnamed_addr #0 {
entry:
;
; CHECK:       ********** MI Scheduling **********
; CHECK_A9:    VADDS
; CHECK_SWIFT: VADDfd
; CHECK_R52:   VADDS
; CHECK_A9:    Latency    : 5
; CHECK_SWIFT: Latency    : 4
; CHECK_R52:   Latency    : 6
;
; CHECK_A9:    VMULS
; CHECK_SWIFT: VMULfd
; CHECK_R52:   VMULS
; CHECK_SWIFT: Latency    : 4
; CHECK_A9:    Latency    : 6
; CHECK_R52:   Latency    : 6
;
; CHECK:       VDIVS
; CHECK_SWIFT: Latency    : 17
; CHECK_A9:    Latency    : 16
; CHECK_R52:   Latency    : 7
;
; CHECK:       VCVTDS
; CHECK_SWIFT: Latency    : 4
; CHECK_A9:    Latency    : 5
; CHECK_R52:   Latency    : 6
;
; CHECK:       VADDD
; CHECK_SWIFT: Latency    : 6
; CHECK_A9:    Latency    : 5
; CHECK_R52:   Latency    : 6
;
; CHECK:       VMULD
; CHECK_SWIFT: Latency    : 6
; CHECK_A9:    Latency    : 7
; CHECK_R52:   Latency    : 6
;
; CHECK:       VDIVD
; CHECK_SWIFT: Latency    : 32
; CHECK_A9:    Latency    : 26
; CHECK_R52:   Latency    : 17
;
; CHECK:       VTOSIZD
; CHECK_SWIFT: Latency    : 4
; CHECK_A9:    Latency    : 5
; CHECK_R52:   Latency    : 6
;
  %add = fadd float %a, %b
  %mul = fmul float %add, %add
  %div = fdiv float %mul, %b
  %conv1 = fpext float %div to double
  %add3 = fadd double %conv1, %conv1
  %mul4 = fmul double %add3, %add3
  %div5 = fdiv double %mul4, %conv1
  %conv6 = fptosi double %div5 to i32
  ret i32 %conv6
}
