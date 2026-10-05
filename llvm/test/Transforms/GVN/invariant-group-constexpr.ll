; RUN: opt < %s -passes=gvn -S | FileCheck %s

; MemDep must not walk the use list of a constant pointer operand to find an
; !invariant.group dependency. A constant GEP into a global is shared by every
; function that uses it, so the walk would find loads and stores in other
; functions and GVN would forward their values across function boundaries.

@flag = global [16 x i32] zeroinitializer
@arr = global [256 x double] zeroinitializer

define void @init() {
  store i32 1, ptr getelementptr inbounds (i8, ptr @flag, i64 4), !invariant.group !0
  ret void
}

; The store in @init must not be forwarded.
define i32 @use_flag(ptr %p) {
; CHECK-LABEL: define i32 @use_flag(
; CHECK:         [[V:%.*]] = load i32, ptr getelementptr inbounds (i8, ptr @flag, i64 4), align 4, !invariant.group
; CHECK:         [[S:%.*]] = add i32 %{{.*}}, [[V]]
; CHECK:         ret i32 [[S]]
  %x = load i32, ptr %p
  %v = load i32, ptr getelementptr inbounds (i8, ptr @flag, i64 4), !invariant.group !0
  %s = add i32 %x, %v
  ret i32 %s
}

; The load in @load_other must not be forwarded into @load_first.
define double @load_first(ptr %p) {
; CHECK-LABEL: define double @load_first(
; CHECK:         [[V:%.*]] = load double, ptr getelementptr inbounds (i8, ptr @arr, i64 16), align 8, !invariant.group
; CHECK:         [[M:%.*]] = fmul double %{{.*}}, [[V]]
; CHECK:         ret double [[M]]
  %x = load double, ptr %p
  %v = load double, ptr getelementptr inbounds (i8, ptr @arr, i64 16), !invariant.group !0
  %m = fmul double %x, %v
  ret double %m
}

define double @load_other() {
; CHECK-LABEL: define double @load_other(
; CHECK:         [[V:%.*]] = load double, ptr getelementptr inbounds (i8, ptr @arr, i64 16), align 8, !invariant.group
; CHECK:         ret double [[V]]
  %v = load double, ptr getelementptr inbounds (i8, ptr @arr, i64 16), !invariant.group !0
  ret double %v
}

!0 = !{}
