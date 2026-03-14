; RUN: opt -S -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@GV = external constant ptr

define ptr @test1(i1 %cond, ptr %P) {
entry:
  br i1 %cond, label %if, label %then

then:
  br label %join

if:
  %load = load ptr, ptr @GV, align 8, !dereferenceable !0
  br label %join

join:
  %phi = phi ptr [ %P, %then ], [ %load, %if ]
  ret ptr %phi
}

; CHECK-LABEL: define ptr @test1(
; CHECK: %[[load:.*]] = load ptr, ptr @GV, align 8{{$}}
; CHECK: %[[phi:.*]] = select i1 %cond, ptr %[[load]], ptr %P
; CHECK: ret ptr %[[phi]]


!0 = !{i64 8}
