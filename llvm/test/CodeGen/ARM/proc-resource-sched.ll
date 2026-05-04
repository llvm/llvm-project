; RUN: llc -mtriple=arm-eabi -mcpu=cortex-r52 -debug-only=machine-scheduler %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-R52
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-r52plus -debug-only=machine-scheduler %s -o - 2>&1 | FileCheck %s --check-prefix=CHECK-R52
; REQUIRES: asserts

; source_filename = "sched-2.c"
target datalayout = "e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64"

define dso_local i32 @f(i32 %a, i32 %b, i32 %c, i32 %d) local_unnamed_addr {
entry:
  %add = add nsw i32 %b, %a
  %add1 = add nsw i32 %d, %c
  %div = sdiv i32 %add, %add1
  ret i32 %div
}

; Cortex-R52 model describes it as dual-issue with two integer ALUs
; It should be able to issue the two additions in the same cycle.
; CHECK-R52: MI Scheduling
; CHECK-R52: Cycle: 14
; CHECK-R52: Scheduling SU(5) %5:gpr = nsw ADDrr %3:gpr, %2:gpr, 14, $noreg, $noreg
; CHECK-R52: Scheduling SU(4) %4:gpr = nsw ADDrr %1:gpr, %0:gpr, 14, $noreg, $noreg
; CHECK-R52: Cycle: 15
