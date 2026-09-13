; RUN: opt %s -passes=slp-vectorizer -S -mtriple=x86_64-- \
; RUN:   -mcpu=skylake-avx512 | FileCheck %s --check-prefix=STLF
; RUN: opt %s -passes=slp-vectorizer -S -mtriple=x86_64-- \
; RUN:   -mcpu=skylake-avx512 -slp-store-load-forward-check=false \
; RUN:   | FileCheck %s --check-prefix=NO-STLF

; A widened load reads bytes produced by multiple sparse stores. SLP represents
; the stores as an expanded masked store, so they remain narrow writes for
; forwarding purposes and the load-side STLF cost must still apply.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @widened_load_with_expanded_store(ptr noalias %A, i64 %n) {
; STLF-LABEL: define void @widened_load_with_expanded_store(
; STLF-COUNT-4: load i32
; STLF-COUNT-4: store i32
; STLF-NOT: llvm.masked.store
;
; NO-STLF-LABEL: define void @widened_load_with_expanded_store(
; NO-STLF: load <4 x i32>
; NO-STLF: llvm.masked.store
entry:
  br label %loop

loop:
  %i = phi i64 [ 8, %entry ], [ %next, %loop ]
  %b0 = add i64 %i, -4
  %b1 = add i64 %i, -3
  %b2 = add i64 %i, -2
  %b3 = add i64 %i, -1
  %p0 = getelementptr inbounds i32, ptr %A, i64 %b0
  %p1 = getelementptr inbounds i32, ptr %A, i64 %b1
  %p2 = getelementptr inbounds i32, ptr %A, i64 %b2
  %p3 = getelementptr inbounds i32, ptr %A, i64 %b3
  %l0 = load i32, ptr %p0, align 4
  %l1 = load i32, ptr %p1, align 4
  %l2 = load i32, ptr %p2, align 4
  %l3 = load i32, ptr %p3, align 4

  %s2 = add i64 %i, 2
  %s4 = add i64 %i, 4
  %s6 = add i64 %i, 6
  %q0 = getelementptr inbounds i32, ptr %A, i64 %i
  %q1 = getelementptr inbounds i32, ptr %A, i64 %s2
  %q2 = getelementptr inbounds i32, ptr %A, i64 %s4
  %q3 = getelementptr inbounds i32, ptr %A, i64 %s6
  store i32 %l0, ptr %q0, align 4
  store i32 %l1, ptr %q1, align 4
  store i32 %l2, ptr %q2, align 4
  store i32 %l3, ptr %q3, align 4

  %next = add nuw nsw i64 %i, 4
  %done = icmp uge i64 %next, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
