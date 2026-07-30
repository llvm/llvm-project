; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%class.A = type { ptr }

@0 = constant [1 x ptr] zeroinitializer

@vtbl = alias ptr, ptr @0

define ptr @test(i1 %c1, i1 %c2) {
; CHECK-LABEL: test
entry:
  br i1 %c1, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  br i1 %c2, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  %A = phi ptr [ @vtbl, %for.body ], [ null, %entry ]
  %B = load ptr, ptr %A
  ret ptr %B
}
