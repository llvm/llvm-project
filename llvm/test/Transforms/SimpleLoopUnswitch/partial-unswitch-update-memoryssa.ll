; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial>),verify<loops>' -verify-dom-info -verify-memoryssa -S %s | FileCheck %s
; RUN: opt -passes='loop-mssa(simple-loop-unswitch<nontrivial>),verify<loops>' -memssa-check-limit=3 -verify-dom-info -verify-memoryssa -S %s | FileCheck %s

declare void @clobber()

; Check that MemorySSA updating can deal with a clobbering access of a
; duplicated load being a MemoryPHI outside the loop.
define void @partial_unswitch_memssa_update(i32* noalias %ptr, i1 %c) {
; CHECK-LABEL: @partial_unswitch_memssa_update(
; CHECK-LABEL: loop.ph:
; CHECK-NEXT:    [[LV:%[a-z0-9]+]] = load i32, i32* %ptr, align 4
; CHECK-NEXT:    [[C:%[a-z0-9]+]] = icmp eq i32 [[LV]], 0
; CHECK-NEXT:    br i1 [[C]]
entry:
  br i1 %c, label %loop.ph, label %outside.clobber

outside.clobber:
  call void @clobber()
  br label %loop.ph

loop.ph:
  br label %loop.header

loop.header:
  %lv = load i32, i32* %ptr, align 4
  %hc = icmp eq i32 %lv, 0
  br i1 %hc, label %if, label %then

if:
  br label %loop.latch

then:
  br label %loop.latch

loop.latch:
  br i1 true, label %loop.header, label %exit

exit:
  ret void
}

; Check that MemorySSA updating can deal with skipping defining accesses in the
; loop body until it finds the first defining access outside the loop.
define void @partial_unswitch_inloop_stores_beteween_outside_defining_access(i64* noalias %ptr, i16* noalias %src) {
; CHECK-LABEL: @partial_unswitch_inloop_stores_beteween_outside_defining_access
; CHECK-LABEL: entry:
; CHECK-NEXT:    store i64 0, i64* %ptr, align 1
; CHECK-NEXT:    store i64 1, i64* %ptr, align 1
; CHECK-NEXT:    [[LV:%[a-z0-9]+]] = load i16, i16* %src, align 1
; CHECK-NEXT:    [[C:%[a-z0-9]+]] = icmp eq i16 [[LV]], 0
; CHECK-NEXT:    br i1 [[C]]
;
entry:
  store i64 0, i64* %ptr, align 1
  store i64 1, i64* %ptr, align 1
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop.latch ]
  store i64 2, i64* %ptr, align 1
  %lv = load i16, i16* %src, align 1
  %invar.cond = icmp eq i16 %lv, 0
  br i1 %invar.cond, label %noclobber, label %loop.latch

noclobber:
  br label %loop.latch

loop.latch:
  %iv.next = add i32 %iv, 1
  %ec = icmp eq i32 %iv, 1000
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

