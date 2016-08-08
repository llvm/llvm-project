; RUN: llc -verify-machineinstrs < %s -mcpu=ppc | FileCheck %s

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-unknown-linux-gnu"

define i64 @foo(i64* nocapture %n, i64 %d) nounwind readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi i64 [ 0, %entry ], [ %conv1, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %n, i32 %i.06
  %0 = load i64, i64* %arrayidx, align 8
  %conv = udiv i64 %x.05, %d
  %conv1 = add i64 %conv, %0
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i64 %conv1
}

; CHECK: @foo
; CHECK-NOT: mtctr

define i64 @foo2(i64* nocapture %n, i64 %d) nounwind readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi i64 [ 0, %entry ], [ %conv1, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %n, i32 %i.06
  %0 = load i64, i64* %arrayidx, align 8
  %conv = sdiv i64 %x.05, %d
  %conv1 = add i64 %conv, %0
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i64 %conv1
}

; CHECK: @foo2
; CHECK-NOT: mtctr

define i64 @foo3(i64* nocapture %n, i64 %d) nounwind readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi i64 [ 0, %entry ], [ %conv1, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %n, i32 %i.06
  %0 = load i64, i64* %arrayidx, align 8
  %conv = urem i64 %x.05, %d
  %conv1 = add i64 %conv, %0
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i64 %conv1
}

; CHECK: @foo3
; CHECK-NOT: mtctr

define i64 @foo4(i64* nocapture %n, i64 %d) nounwind readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi i64 [ 0, %entry ], [ %conv1, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %n, i32 %i.06
  %0 = load i64, i64* %arrayidx, align 8
  %conv = srem i64 %x.05, %d
  %conv1 = add i64 %conv, %0
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret i64 %conv1
}

; CHECK: @foo4
; CHECK-NOT: mtctr

