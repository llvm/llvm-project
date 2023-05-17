; RUN: opt -S -loop-reduce < %s | FileCheck %s

target triple = "x86_64-unknown-unknown"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

define void @incorrect_offset_scaling(i64, ptr) {
top:
  br label %L

L:                                                ; preds = %idxend.10, %idxend, %L2, %top
  br i1 undef, label %L, label %L1

L1:                                               ; preds = %L1.preheader, %L2
  %r13 = phi i64 [ %r1, %L2 ], [ 1, %L ]
; CHECK:  %lsr.iv = phi i64 [ 0, %L{{[^ ]+}} ], [ %lsr.iv.next, %L2 ]
; CHECK-NOT:  %lsr.iv = phi i64 [ -1, %L{{[^ ]+}} ], [ %lsr.iv.next, %L2 ]
; CHECK:  br
  %r0 = add i64 %r13, -1
  br label %idxend.8

L2:                                               ; preds = %idxend.8
  %r1 = add i64 %r13, 1
  br i1 undef, label %L, label %L1

if6:                                              ; preds = %idxend.8
  %r2 = add i64 %0, -1
  %r3 = load i64, ptr %1, align 8
; CHECK:  %r2 = add i64 %0, -1
; CHECK:  %r3 = load i64
  br label %ib

idxend.8:                                         ; preds = %L1
  br i1 undef, label %if6, label %L2

ib:                                               ; preds = %if6
  %r4 = mul i64 %r3, %r0
  %r5 = add i64 %r2, %r4
  %r6 = icmp ult i64 %r5, undef
; CHECK:  %r4 = mul i64 %r3, %lsr.iv
; CHECK:  %r5 = add i64 %r2, %r4
; CHECK:  %r6 = icmp ult i64 %r5, undef
; CHECK:  %r7 = getelementptr i64, ptr undef, i64 %r5
  %r7 = getelementptr i64, ptr undef, i64 %r5
  store i64 1, ptr %r7, align 8
  br label %L
}
