; RUN: opt -analyze -scalar-evolution < %s  -o - -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Test that SCEV is capable of figuring out value of 'IV' that actually does not change.
; CHECK: Classifying expressions for: @foo
; CHECK: %iv.i = phi i64
; CHECK: -5 U: [-5,-4) S: [-5,-4)         Exits: -5               LoopDispositions: { %loop: Invariant }
define void @foo() {
entry:
  br label %loop

loop:
  %iv.i = phi i64 [ -5, %entry ], [ %iv.next.i, %loop ]
  %iv.next.i = add nsw i64 %iv.i, 0
  br label %loop
}
