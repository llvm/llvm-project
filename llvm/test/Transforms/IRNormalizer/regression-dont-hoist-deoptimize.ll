; RUN: opt %s -passes=normalize -verify-each
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128-ni:1-p2:32:8:8:32-ni:2"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.experimental.deoptimize.isVoid(...)

define void @widget() {
bb:
  %tmp3 = trunc i64 0 to i32
  call void (...) @llvm.experimental.deoptimize.isVoid(i32 0) [ "deopt"() ]
  ret void
}
