; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-r52 -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=R52_SCHED
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-r52plus -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=R52_SCHED
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=generic    -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=GENERIC
;
; Check the latency for instructions for both generic and cortex-r52.
; Cortex-r52 machine model will cause the div to be sceduled before eor
; as div takes more cycles to compute than eor.
;
; CHECK:       ********** MI Scheduling **********
; CHECK:      foo:%bb.0 entry
; CHECK:      EORrr
; GENERIC:    Latency    : 1
; R52_SCHED:  Latency    : 3
; CHECK:      MLA
; GENERIC:    Latency    : 2
; R52_SCHED:  Latency    : 4
; CHECK:      SDIV
; GENERIC:    Latency    : 0
; R52_SCHED:  Latency    : 8
; CHECK:      ** Final schedule for %bb.0 ***
; GENERIC:    EORrr
; GENERIC:    SDIV
; R52_SCHED:  SDIV
; R52_SCHED:  EORrr
; CHECK:      ********** INTERVALS **********

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8r-arm-none-eabi"

; Function Attrs: norecurse nounwind readnone
define hidden i32 @foo(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
entry:
  %xor = xor i32 %c, %b
  %mul = mul nsw i32 %xor, %c
  %add = add nsw i32 %mul, %a
  %div = sdiv i32 %a, %b
  %sub = sub i32 %add, %div
  ret i32 %sub
}
