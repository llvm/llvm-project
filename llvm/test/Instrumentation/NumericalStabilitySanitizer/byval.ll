; RUN: opt -passes=nsan -nsan-shadow-type-mapping=dqq -S %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"

; The copy of a byval argument is fresh stack memory with no shadow written by
; the caller. The caller passes the source addresses of its byval arguments, and
; the callee initializes the shadow of its copies from them, in argument order.

%struct.S8 = type { double }
%struct.S24 = type { double, double, double }

define double @use(ptr noundef byval(%struct.S24) align 8 %a) sanitize_numerical_stability {
; CHECK-LABEL: @use(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @__nsan_copy_byval_shadow(ptr %a, i64 24, i64 0, ptr @use)
; CHECK-NEXT:    store i64 0, ptr @__nsan_byval_tag
entry:
  %v = load double, ptr %a, align 8
  ret double %v
}

define double @use_many(i32 %n, ptr byval(%struct.S8) align 8 %a, double %x, ptr byval(%struct.S24) align 8 %b) sanitize_numerical_stability {
; CHECK-LABEL: @use_many(
; CHECK:         call void @__nsan_copy_byval_shadow(ptr %a, i64 8, i64 1, ptr @use_many)
; CHECK-NEXT:    call void @__nsan_copy_byval_shadow(ptr %b, i64 24, i64 3, ptr @use_many)
; CHECK-NEXT:    store i64 0, ptr @__nsan_byval_tag
entry:
  %v = load double, ptr %b, align 8
  ret double %v
}

; A function without byval arguments does not touch the byval tag.
define double @no_byval(double %x) sanitize_numerical_stability {
; CHECK-LABEL: @no_byval(
; CHECK-NOT:     __nsan_byval_tag
entry:
  ret double %x
}

define double @call_use(ptr %p) sanitize_numerical_stability {
; CHECK-LABEL: @call_use(
; CHECK:         store ptr @use, ptr @__nsan_byval_tag
; CHECK-NEXT:    store ptr %p, ptr @__nsan_byval_ptrs
; CHECK-NEXT:    %r = call double @use(ptr byval(%struct.S24) align 8 %p)
entry:
  %r = call double @use(ptr byval(%struct.S24) align 8 %p)
  ret double %r
}

; Only byval arguments get a pointer slot, indexed by argument position, and
; the FP shadow stack is only set up when there are FP arguments.
define double @call_use_many(ptr %p, ptr %q) sanitize_numerical_stability {
; CHECK-LABEL: @call_use_many(
; CHECK:         store ptr @use_many, ptr @__nsan_byval_tag
; CHECK-NEXT:    store ptr %p, ptr getelementptr (i8, ptr @__nsan_byval_ptrs, i64 8)
; CHECK-NEXT:    store ptr %q, ptr getelementptr (i8, ptr @__nsan_byval_ptrs, i64 24)
; CHECK-NEXT:    store ptr @use_many, ptr @__nsan_shadow_args_tag
entry:
  %r = call double @use_many(i32 1, ptr byval(%struct.S8) align 8 %p, double 2.0, ptr byval(%struct.S24) align 8 %q)
  ret double %r
}
