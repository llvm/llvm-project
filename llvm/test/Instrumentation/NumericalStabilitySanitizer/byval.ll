; RUN: opt -passes=nsan -nsan-shadow-type-mapping=dqq -S %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

; The copy of a byval argument is fresh stack memory with no shadow written by
; the caller, so the callee must reset its shadow to unknown. The calls must
; appear in argument order and use the size-specialized runtime functions.

%struct.S4 = type { float }
%struct.S8 = type { double }
%struct.S16 = type { double, double }
%struct.S24 = type { double, double, double }

define double @use(ptr noundef byval(%struct.S24) align 8 %a) sanitize_numerical_stability {
; CHECK-LABEL: @use(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__nsan_set_value_unknown(ptr %a, i64 24)
entry:
  %v = load double, ptr %a, align 8
  ret double %v
}

define double @use_many(ptr byval(%struct.S4) align 4 %a, ptr byval(%struct.S8) align 8 %b, ptr byval(%struct.S16) align 8 %c, ptr byval(%struct.S24) align 8 %d) sanitize_numerical_stability {
; CHECK-LABEL: @use_many(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__nsan_set_value_unknown_4(ptr %a)
; CHECK-NEXT:    call void @__nsan_set_value_unknown_8(ptr %b)
; CHECK-NEXT:    call void @__nsan_set_value_unknown_16(ptr %c)
; CHECK-NEXT:    call void @__nsan_set_value_unknown(ptr %d, i64 24)
entry:
  %v = load double, ptr %b, align 8
  ret double %v
}
