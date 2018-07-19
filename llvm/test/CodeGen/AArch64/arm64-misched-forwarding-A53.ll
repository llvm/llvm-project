; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a53 -pre-RA-sched=source -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
;
; For Cortex-A53, shiftable operands that are not actually shifted
; are not needed for an additional two cycles.
;
; CHECK: ********** MI Scheduling **********
; CHECK: shiftable
; CHECK: SU(2):   %2:gpr64common = SUBXri %1, 20, 0
; CHECK:   Successors:
; CHECK-NEXT:    SU(4): Data Latency=1 Reg=%2
; CHECK-NEXT:    SU(3): Data Latency=2 Reg=%2
; CHECK: ********** INTERVALS **********
define i64 @shiftable(i64 %A, i64 %B) {
        %tmp0 = sub i64 %B, 20
        %tmp1 = shl i64 %tmp0, 5;
        %tmp2 = add i64 %A, %tmp1;
        %tmp3 = add i64 %A, %tmp0
        %tmp4 = mul i64 %tmp2, %tmp3

        ret i64 %tmp4
}
