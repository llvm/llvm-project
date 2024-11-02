; RUN: opt -O2 -S %s | FileCheck %s
; RUN: opt -passes="default<O2>" -S %s | FileCheck %s

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; The loop below needs to be peeled first to eliminate the constant PHI %first
; before loop vectorization.
;
; Test case from PR47671.

define i32 @test(i32* readonly %p, i32* readnone %q) {
; CHECK-LABEL: define i32 @test(
; CHECK: vector.body:
; CHECK:   %index.next = add nuw i64 %index, 8
; CHECK: middle.block:
;
entry:
  %cmp.not7 = icmp eq i32* %p, %q
  br i1 %cmp.not7, label %exit, label %loop.ph

loop.ph:
  br label %loop

loop:
  %sum = phi i32 [ %sum.next, %loop ], [ 0, %loop.ph ]
  %first = phi i1 [ false, %loop ], [ true, %loop.ph ]
  %iv = phi i32* [ %iv.next, %loop ], [ %p, %loop.ph ]
  %add = add nsw i32 %sum, 2
  %spec.select = select i1 %first, i32 %sum, i32 %add
  %lv = load i32, i32* %iv, align 4
  %sum.next = add nsw i32 %lv, %spec.select
  %iv.next = getelementptr inbounds i32, i32* %iv, i64 1
  %cmp.not = icmp eq i32* %iv.next, %q
  br i1 %cmp.not, label %loopexit, label %loop

loopexit:
  %sum.next.lcssa = phi i32 [ %sum.next, %loop ]
  br label %exit

exit:
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %sum.next.lcssa, %loopexit ]
  ret i32 %sum.0.lcssa
}
