; RUN: opt -S -simplifycfg < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@GV = external constant i64*

define i64* @test1(i1 %cond, i8* %P) {
entry:
  br i1 %cond, label %if, label %then

then:
  %bc = bitcast i8* %P to i64*
  br label %join

if:
  %load = load i64*, i64** @GV, align 8, !dereferenceable !0
  br label %join

join:
  %phi = phi i64* [ %bc, %then ], [ %load, %if ]
  ret i64* %phi
}

; CHECK-LABEL: define i64* @test1(
; CHECK: %[[bc:.*]] = bitcast i8* %P to i64*
; CHECK: %[[load:.*]] = load i64*, i64** @GV, align 8{{$}}
; CHECK: %[[phi:.*]] = select i1 %cond, i64* %[[load]], i64* %[[bc]]
; CHECK: ret i64* %[[phi]]


!0 = !{i64 8}
