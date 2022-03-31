; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%class.A = type { i32 (...)** }

@0 = constant [1 x i8*] zeroinitializer

@vtbl = alias i8*, getelementptr inbounds ([1 x i8*], [1 x i8*]* @0, i32 0, i32 0)

define i32 (%class.A*)* @test() {
; CHECK-LABEL: test
entry:
  br i1 undef, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  br i1 undef, label %for.body, label %for.end

for.end:                                          ; preds = %for.body, %entry
  %A = phi i32 (%class.A*)** [ bitcast (i8** @vtbl to i32 (%class.A*)**), %for.body ], [ null, %entry ]
  %B = load i32 (%class.A*)*, i32 (%class.A*)** %A
  ret i32 (%class.A*)* %B
}
