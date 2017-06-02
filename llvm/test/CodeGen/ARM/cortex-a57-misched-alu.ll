; REQUIRES: asserts
; RUN: llc < %s -mtriple=armv8r-eabi -mcpu=cortex-a57 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s

; Check the latency for ALU shifted operand variants.
;
; CHECK:       ********** MI Scheduling **********
; CHECK:      foo:BB#0 entry

; ALU, basic - 1 cyc I0/I1
; CHECK:      EORrr
; CHECK:      rdefs left
; CHECK-NEXT: Latency    : 1

; ALU, shift by immed - 2 cyc M
; CHECK:      ADDrsi
; CHECK:      rdefs left
; CHECK-NEXT: Latency    : 2

; ALU, shift by register, unconditional - 2 cyc M
; CHECK:      RSBrsr
; CHECK:      rdefs left
; CHECK-NEXT: Latency    : 2

; ALU, shift by register, conditional - 2 cyc I0/I1
; CHECK:      ANDrsr
; CHECK:      rdefs left
; CHECK-NEXT: Latency    : 2

; Checking scheduling units

; CHECK:      ** ScheduleDAGMILive::schedule picking next node
; Skipping COPY
; CHECK:      ** ScheduleDAGMILive::schedule picking next node
; CHECK:      Scheduling
; CHECK-SAME: ANDrsr
; CHECK:      Ready
; CHECK-NEXT: A57UnitI

; CHECK:      ** ScheduleDAGMILive::schedule picking next node
; CHECK:      Scheduling
; CHECK-SAME: CMPri
; CHECK:      Ready
; CHECK-NEXT: A57UnitI

; CHECK:      ** ScheduleDAGMILive::schedule picking next node
; CHECK:      Scheduling
; CHECK-SAME: RSBrsr
; CHECK:      Ready
; CHECK-NEXT: A57UnitM

; CHECK:      ** ScheduleDAGMILive::schedule picking next node
; CHECK:      Scheduling
; CHECK-SAME: ADDrsi
; CHECK:      Ready
; CHECK-NEXT: A57UnitM

; CHECK:      ** ScheduleDAGMILive::schedule picking next node
; CHECK:      Scheduling
; CHECK-SAME: EORrr
; CHECK:      Ready
; CHECK-NEXT: A57UnitI


target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv8r-arm-none-eabi"

; Function Attrs: norecurse nounwind readnone
define hidden i32 @foo(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr #0 {
entry:
  %xor = xor i32 %a, %b
  %xor_shl = shl i32 %xor, 2
  %add = add i32 %xor_shl, %d
  %add_ashr = ashr i32 %add, %a
  %sub = sub i32 %add_ashr, %a
  %sub_lshr_pred = lshr i32 %sub, %c
  %pred = icmp sgt i32 %a, 4
  %and = and i32 %sub_lshr_pred, %b
  %rv = select i1 %pred, i32 %and, i32 %d
  ret i32 %rv
}

