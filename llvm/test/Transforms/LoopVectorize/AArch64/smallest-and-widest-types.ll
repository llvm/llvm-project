; REQUIRES: asserts
; RUN: opt < %s -loop-vectorize -debug-only=loop-vectorize -disable-output 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

; CHECK-LABEL: Checking a loop in "interleaved_access"
; CHECK:         The Smallest and Widest types: 64 / 64 bits
;
define void @interleaved_access(i8** %A, i64 %N) {
for.ph:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next.3, %for.body ], [ 0, %for.ph ]
  %tmp0 = getelementptr inbounds i8*, i8** %A, i64 %i
  store i8* null, i8** %tmp0, align 8
  %i.next.0 = add nuw nsw i64 %i, 1
  %tmp1 = getelementptr inbounds i8*, i8** %A, i64 %i.next.0
  store i8* null, i8** %tmp1, align 8
  %i.next.1 = add nsw i64 %i, 2
  %tmp2 = getelementptr inbounds i8*, i8** %A, i64 %i.next.1
  store i8* null, i8** %tmp2, align 8
  %i.next.2 = add nsw i64 %i, 3
  %tmp3 = getelementptr inbounds i8*, i8** %A, i64 %i.next.2
  store i8* null, i8** %tmp3, align 8
  %i.next.3 = add nsw i64 %i, 4
  %cond = icmp slt i64 %i.next.3, %N
  br i1 %cond, label %for.body, label %for.end

for.end:
  ret void
}
