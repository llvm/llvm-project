; RUN: opt -passes=loop-distribute -enable-loop-distribute -verify-loop-info -verify-dom-info -S \
; RUN:   < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @fn1(i64 %a, i64* %b) {
entry:
  br label %for.body

for.body:
  %add75.epil = phi i64 [ %add7.epil, %for.body ], [ %a, %entry ]
  %add1.epil = add nsw i64 %add75.epil, 268435457
  %arrayidx.epil = getelementptr inbounds i64, i64* %b, i64 %add1.epil
  %load = load i64, i64* %arrayidx.epil, align 8
  %add5.epil = add nsw i64 %add75.epil, 805306369
  %arrayidx6.epil = getelementptr inbounds i64, i64* %b, i64 %add5.epil
  store i64 %load, i64* %arrayidx6.epil, align 8
  %add7.epil = add nsw i64 %add75.epil, 2
  %epil.iter.cmp = icmp eq i64 %add7.epil, 0
  br i1 %epil.iter.cmp, label %for.end, label %for.body

  ; CHECK: %[[phi:.*]]  = phi i64
  ; CHECK: %[[add1:.*]] = add nsw i64 %[[phi]], 268435457
  ; CHECK: %[[gep1:.*]] = getelementptr inbounds i64, i64* %b, i64 %[[add1]]
  ; CHECK: %[[load:.*]] = load i64, i64* %[[gep1]], align 8
  ; CHECK: %[[add2:.*]] = add nsw i64 %[[phi]], 805306369
  ; CHECK: %[[gep2:.*]] = getelementptr inbounds i64, i64* %b, i64 %[[add2]]
  ; CHECK: store i64 %[[load]], i64* %[[gep2]], align 8
  ; CHECK: %[[incr:.*]] = add nsw i64 %[[phi]], 2
  ; CHECK: %[[cmp:.*]]  = icmp eq i64 %[[incr]], 0
  ; CHECK: br i1 %[[cmp]]

for.end:
  ret void
}
