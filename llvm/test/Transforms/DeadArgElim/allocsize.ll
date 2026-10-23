; RUN: opt < %s -passes=deadargelim -S | FileCheck %s
; PR36867

; Deleting an argument ahead of the one allocsize names renumbers it, so the
; attribute has to follow rather than be dropped.

; CHECK-LABEL: define internal i64 @MagickMallocAligned(i64 %s)
; CHECK-SAME: #[[ONE:[0-9]+]]
define internal i64 @MagickMallocAligned(i64 %DEADARG1, i64 %s) allocsize(1) {
        ret i64 %s
}

define i64 @NeedsArg(i64 %s) {
; CHECK-LABEL: define i64 @NeedsArg(
; CHECK: call i64 @MagickMallocAligned(i64 %s)
	%c = call i64 @MagickMallocAligned(i64 0, i64 %s)
	ret i64 %c
}

define i64 @Test2(i64 %s) {
; CHECK-LABEL: define i64 @Test2(
; CHECK: call i64 @MagickMallocAligned(i64 %s) #[[ONE]]
	%c = call i64 @MagickMallocAligned(i64 0, i64 %s) allocsize(1)
	ret i64 %c
}

; Both indices of a calloc-like allocsize are renumbered.

; CHECK-LABEL: define internal ptr @two_args(i64 %n, i64 %sz)
; CHECK-SAME: #[[TWO:[0-9]+]]
define internal ptr @two_args(i64 %DEADARG2, i64 %n, i64 %sz) allocsize(1, 2) {
  %p = call ptr @allocate(i64 %n, i64 %sz)
  ret ptr %p
}

define ptr @calls_two_args(i64 %n, i64 %sz) {
  %p = call ptr @two_args(i64 0, i64 %n, i64 %sz)
  ret ptr %p
}

; The argument allocsize names is otherwise unused, but removing it would cost
; the attribute, so it is kept alive.

; CHECK-LABEL: define internal ptr @unused_size(i64 %n)
; CHECK-SAME: #[[THREE:[0-9]+]]
define internal ptr @unused_size(i64 %DEADARG3, i64 %n) allocsize(1) nounwind {
  %p = call ptr @allocate(i64 0, i64 0)
  ret ptr %p
}

define ptr @calls_unused_size(i64 %n) {
  %p = call ptr @unused_size(i64 0, i64 %n)
  ret ptr %p
}

; A call site's allocsize names an argument of a callee that has no allocsize of
; its own and never uses it. Keeping the attribute means keeping the argument.

; CHECK-LABEL: define internal ptr @callsite_names_size(i64 %sz)
define internal ptr @callsite_names_size(i64 %DEADARG4, i64 %sz) {
  %p = call ptr @allocate(i64 0, i64 0)
  ret ptr %p
}

define ptr @calls_callsite_names_size(i64 %sz) {
; CHECK-LABEL: define ptr @calls_callsite_names_size(
; CHECK: call ptr @callsite_names_size(i64 %sz) #[[ONE]]
  %p = call ptr @callsite_names_size(i64 0, i64 %sz) allocsize(1)
  ret ptr %p
}

declare ptr @allocate(i64, i64)

; CHECK-DAG: attributes #[[ONE]] = { allocsize(0) }
; CHECK-DAG: attributes #[[TWO]] = { allocsize(0,1) }
; CHECK-DAG: attributes #[[THREE]] = { nounwind allocsize(0) }
