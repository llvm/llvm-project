; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -enable-misched -verify-misched -debug-only=misched -o - 2>&1 > /dev/null | FileCheck %s
;
; Test for bug in misched memory dependency calculation.
;
; CHECK: ********** MI Scheduling **********
; CHECK: misched_bug:BB#0 entry
; CHECK: SU(2):   %vreg2<def> = LDRWui %vreg0, 1; mem:LD4[%ptr1_plus1] GPR32:%vreg2 GPR64common:%vreg0
; CHECK:   Successors:
; CHECK-NEXT:    val SU(5): Latency=4 Reg=%vreg2
; CHECK-NEXT:    ch  SU(4): Latency=0
; CHECK: SU(3):   STRWui %WZR, %vreg0, 0; mem:ST4[%ptr1] GPR64common:%vreg0
; CHECK:   Successors:
; CHECK: ch  SU(4): Latency=0
; CHECK: SU(4):   STRWui %WZR, %vreg1, 0; mem:ST4[%ptr2] GPR64common:%vreg1
; CHECK: SU(5):   %W0<def> = COPY %vreg2; GPR32:%vreg2
; CHECK: ** ScheduleDAGMI::schedule picking next node
define i32 @misched_bug(i32* %ptr1, i32* %ptr2) {
entry:
  %ptr1_plus1 = getelementptr inbounds i32, i32* %ptr1, i64 1
  %val1 = load i32, i32* %ptr1_plus1, align 4
  store i32 0, i32* %ptr1, align 4
  store i32 0, i32* %ptr2, align 4
  ret i32 %val1
}
