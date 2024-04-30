; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=A57_SCHED
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=generic    -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s --check-prefix=CHECK --check-prefix=GENERIC

; Check the latency for instructions for both generic and cortex-a57.
; SDIV should be scheduled at the block's begin (20 cyc of independent M unit).
;
; CHECK:       ********** MI Scheduling **********
; CHECK:      foo:%bb.0 entry

; GENERIC:    LDRi12
; GENERIC:    Latency    : 1
; GENERIC:    EORrr
; GENERIC:    Latency    : 1
; GENERIC:    ADDrr
; GENERIC:    Latency    : 1
; GENERIC:    SDIV
; GENERIC:    Latency    : 0
; GENERIC:    SUBrr
; GENERIC:    Latency    : 1

; A57_SCHED:  SDIV
; A57_SCHED:  Latency    : 20
; A57_SCHED:  EORrr
; A57_SCHED:  Latency    : 1
; A57_SCHED:  LDRi12
; A57_SCHED:  Latency    : 4
; A57_SCHED:  ADDrr
; A57_SCHED:  Latency    : 1
; A57_SCHED:  SUBrr
; A57_SCHED:  Latency    : 1

; CHECK:      ** Final schedule for %bb.0 ***
; GENERIC:    LDRi12
; GENERIC:    SDIV
; A57_SCHED:  SDIV
; A57_SCHED:  LDRi12
; CHECK:      ********** INTERVALS **********

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8r-arm-none-eabi"

; Function Attrs: norecurse nounwind readnone
define hidden i32 @foo(i32 %a, i32 %b, i32 %c, ptr %d) local_unnamed_addr #0 {
entry:
  %xor = xor i32 %c, %b
  %ld = load i32, ptr %d
  %add = add nsw i32 %xor, %ld
  %div = sdiv i32 %a, %b
  %sub = sub i32 %div, %add
  ret i32 %sub
}

