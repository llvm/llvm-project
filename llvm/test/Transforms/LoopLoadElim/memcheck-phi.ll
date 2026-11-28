; RUN: opt -passes=loop-load-elim -S < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Check that an intervening store whose pointer operand is a non-header PHI
; inside the loop is correctly recognized as an intervening store on the
; forwarding path, requiring a runtime alias check and loop versioning.
define i8 @phi_intervening_store(ptr %a, ptr %b) {
; CHECK-LABEL: define i8 @phi_intervening_store(
; CHECK-SAME: ptr %a, ptr %b) {
; CHECK-NEXT:  loop.lver.check:
; CHECK:         %found.conflict = and i1 %bound0, %bound1
; CHECK-NEXT:    br i1 %found.conflict, label %loop.ph.lver.orig, label %loop.ph
; CHECK:       loop.lver.orig:
; CHECK:         %x.lver.orig = load i8, ptr %pa.lver.orig, align 1
; CHECK:       loop.ph:
; CHECK:         %store_forwarded = phi i8 [ %load_initial, %loop.ph ], [ 0, %m ]
; CHECK:       exit:
; CHECK:         ret i8 %x.lcssa
entry:
  br label %loop

loop:                                             ; preds = %m, %entry
  %i1 = phi i64 [ 0, %entry ], [ %i.next, %m ]
  %pa = getelementptr i8, ptr %a, i64 %i1
  %x = load i8, ptr %pa, align 1
  %i.next = add i64 %i1, 1
  %pa1 = getelementptr i8, ptr %a, i64 %i.next
  store i8 0, ptr %pa1, align 1
  br label %m

m:                                                ; preds = %loop
  %pb = phi ptr [ %b, %loop ]
  store i8 1, ptr %pb, align 1
  %done = icmp eq i64 %i1, 1
  br i1 %done, label %exit, label %loop

exit:                                             ; preds = %m
  ret i8 %x
}

; Check that when a candidate load accesses a pointer that is also written to
; in the loop (making PointerInfo.IsWritePtr true), the load is still matched as
; a candidate load and alias-checked against intervening stores.
define i8 @cand_load_and_store_same_ptr(ptr %a, ptr %b) {
; CHECK-LABEL: define i8 @cand_load_and_store_same_ptr(
; CHECK-SAME: ptr %a, ptr %b) {
; CHECK-NEXT:  loop.lver.check:
; CHECK:         %found.conflict = and i1 %bound0, %bound1
; CHECK-NEXT:    br i1 %found.conflict, label %loop.ph.lver.orig, label %loop.ph
; CHECK:       loop.lver.orig:
; CHECK:         %x.lver.orig = load i8, ptr %pa.lver.orig, align 1
; CHECK:       loop.ph:
; CHECK:         %store_forwarded = phi i8 [ %load_initial, %loop.ph ], [ 0, %loop ]
; CHECK:       exit:
; CHECK:         ret i8
entry:
  br label %loop

loop:                                             ; preds = %loop, %entry
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %pa = getelementptr i8, ptr %a, i64 %i
  %x = load i8, ptr %pa, align 1
  store i8 42, ptr %pa, align 1
  %i.next = add i64 %i, 1
  %pa1 = getelementptr i8, ptr %a, i64 %i.next
  store i8 0, ptr %pa1, align 1
  store i8 1, ptr %b, align 1
  %done = icmp eq i64 %i, 1
  br i1 %done, label %exit, label %loop

exit:                                             ; preds = %loop
  ret i8 %x
}

