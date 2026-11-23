; RUN: opt -passes=nsan -nsan-shadow-type-mapping=dqq -S %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

; The copy of a byval argument is fresh stack memory with no shadow written by
; the caller, so the callee must reset its shadow to unknown.

%struct.S = type { double, double, double }

define double @use(ptr noundef byval(%struct.S) align 8 %a) sanitize_numerical_stability {
; CHECK-LABEL: @use(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__nsan_set_value_unknown(ptr %a, i64 24)
entry:
  %v = load double, ptr %a, align 8
  ret double %v
}
