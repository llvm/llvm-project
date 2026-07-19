; RUN: opt < %s -passes=sink -S | FileCheck %s
;
; A read-only call must not be sunk below an atomic store or RMW with release
; (or stronger) ordering: moving a program-order-earlier read below a release
; operation breaks the publication guarantee. Given a second thread executing
;
;   if (atomic_load_explicit(&flag, memory_order_acquire) == 1)
;     data = 42;
;
; the pre-transform functions below are race-free: the call's read of @data
; happens-before the release operation on @flag, which synchronizes-with the
; acquire load, which happens-before the second thread's store to @data.
; Sinking the call below the release operation would leave the read unordered
; with that store, introducing a data race.
;
; Sinking past a monotonic store remains legal: monotonic ordering imposes no
; constraint on other locations.

@flag = global i32 0
@data = global i32 0

declare i32 @read_data(ptr) nounwind willreturn memory(argmem: read)

define i32 @no_sink_call_past_release_store(i1 %c) {
; CHECK-LABEL: @no_sink_call_past_release_store(
; CHECK:         [[V:%.*]] = call i32 @read_data(ptr @data)
; CHECK-NEXT:    store atomic i32 1, ptr @flag release
entry:
  %v = call i32 @read_data(ptr @data)
  store atomic i32 1, ptr @flag release, align 4
  br i1 %c, label %use, label %skip

use:
  ret i32 %v

skip:
  ret i32 0
}

define i32 @no_sink_call_past_seq_cst_store(i1 %c) {
; CHECK-LABEL: @no_sink_call_past_seq_cst_store(
; CHECK:         [[V:%.*]] = call i32 @read_data(ptr @data)
; CHECK-NEXT:    store atomic i32 1, ptr @flag seq_cst
entry:
  %v = call i32 @read_data(ptr @data)
  store atomic i32 1, ptr @flag seq_cst, align 4
  br i1 %c, label %use, label %skip

use:
  ret i32 %v

skip:
  ret i32 0
}

define i32 @no_sink_call_past_release_rmw(i1 %c) {
; CHECK-LABEL: @no_sink_call_past_release_rmw(
; CHECK:         [[V:%.*]] = call i32 @read_data(ptr @data)
; CHECK-NEXT:    {{%.*}} = atomicrmw add ptr @flag, i32 1 release
entry:
  %v = call i32 @read_data(ptr @data)
  %old = atomicrmw add ptr @flag, i32 1 release, align 4
  br i1 %c, label %use, label %skip

use:
  ret i32 %v

skip:
  ret i32 0
}

; Monotonic imposes no cross-location ordering; the call may be sunk.
define i32 @sink_call_past_monotonic_store(i1 %c) {
; CHECK-LABEL: @sink_call_past_monotonic_store(
; CHECK:       entry:
; CHECK-NEXT:    store atomic i32 1, ptr @flag monotonic
; CHECK:       use:
; CHECK-NEXT:    [[V:%.*]] = call i32 @read_data(ptr @data)
entry:
  %v = call i32 @read_data(ptr @data)
  store atomic i32 1, ptr @flag monotonic, align 4
  br i1 %c, label %use, label %skip

use:
  ret i32 %v

skip:
  ret i32 0
}
