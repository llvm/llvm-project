; RUN: opt -S -passes=indvars < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i16 @fn1(i16 returned, i64)

define void @fn2() {
; CHECK-LABEL: @fn2(
entry:
  br label %for.cond

for.cond:
  %f.0 = phi i64 [ undef, %entry ], [ %inc, %for.cond ]
  %conv = trunc i64 %f.0 to i16
  %call = tail call i16 @fn1(i16 %conv, i64 %f.0)
  %conv2 = zext i16 %call to i32
  %inc = add nsw i64 %f.0, 1
  br label %for.cond
}
