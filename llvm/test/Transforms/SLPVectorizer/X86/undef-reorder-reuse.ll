; RUN: opt -passes=slp-vectorizer -disable-output < %s

; The undef lane matches a poison lane in a reordered, reused TreeEntry. It
; must remain poison rather than being looked up as a concrete scalar lane.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @undef_in_reordered_reuse() {
entry:
  %load.0 = load i8, ptr null, align 1
  %load.1 = load i8, ptr getelementptr inbounds nuw (i8, ptr null, i64 1), align 1
  br label %entry.next

entry.next:
  br i1 false, label %if, label %merge

if:
  %phi.load.1 = phi i8 [ %load.1, %entry.next ]
  %phi.load.0 = phi i8 [ %load.0, %entry.next ]
  %phi.undef = phi i8 [ undef, %entry.next ]
  br label %merge

merge:
  %phi.0 = phi i8 [ %phi.load.1, %if ], [ %load.1, %entry.next ]
  %phi.1 = phi i8 [ %phi.load.0, %if ], [ %load.0, %entry.next ]
  %phi.2 = phi i8 [ %phi.load.1, %if ], [ %load.1, %entry.next ]
  %result = phi i8 [ %phi.undef, %if ], [ poison, %entry.next ]
  ret i8 %result
}
