; RUN: opt -licm -enable-mssa-loop-dependency -S %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @d()
define dso_local void @d() {
entry:
  br label %header

header:
  store i32 1, i32* null, align 4
  br i1 true, label %cleanup53, label %body

body:
  br i1 undef, label %cleanup31, label %for.cond11

for.cond11: ; Needs branch as is
  br i1 undef, label %unreachable, label %latch

cleanup31:
  br label %unreachable

deadblock:
  br i1 undef, label %unreachable, label %deadblock

cleanup53:
  %val = load i32, i32* null, align 4
  %cmpv = icmp eq i32 %val, 0
  br i1 %cmpv, label %cleanup63, label %latch

latch:
  br label %header

cleanup63:
  ret void

unreachable:
  unreachable
}

