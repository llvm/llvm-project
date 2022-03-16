; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@c = external global i32

; CHECK-LABEL: f
; CHECK: MayAlias: i32* %arrayidx, i32* %arrayidx6
define void @f() {
  %idxprom = zext i32 undef to i64
  %add4 = add i32 0, 1
  %idxprom5 = zext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* @c, i64 %idxprom5
  %arrayidx = getelementptr inbounds i32, i32* @c, i64 %idxprom
  load i32, i32* %arrayidx
  load i32, i32* %arrayidx6
  ret void
}

