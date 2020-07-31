; RUN: opt -S -ipsccp < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

define void @test(i32 signext %n) {

; CHECK-LABEL: @test

entry:
  br i1 undef, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  ret void

if.end:                                           ; preds = %entry
  br i1 undef, label %if.then2, label %if.end4

if.then2:                                         ; preds = %if.end
  unreachable

if.end4:                                          ; preds = %if.end
  %sub.n = select i1 undef, i32 undef, i32 %n
  switch i32 %sub.n, label %if.else14 [
    i32 0, label %if.then9
    i32 1, label %if.then12
  ]

if.then9:                                         ; preds = %if.end4
  unreachable

if.then12:                                        ; preds = %if.end4
  unreachable

if.else14:                                        ; preds = %if.end4
  br label %do.body

do.body:                                          ; preds = %do.body, %if.else14
  %scale.0 = phi ppc_fp128 [ 0xM3FF00000000000000000000000000000, %if.else14 ], [ %scale.0, %do.body ]
  br i1 undef, label %do.body, label %if.then33

if.then33:                                        ; preds = %do.body
  br i1 undef, label %_ZN5boost4math4signIgEEiRKT_.exit30, label %cond.false.i28

cond.false.i28:                                   ; preds = %if.then33
  %0 = bitcast ppc_fp128 %scale.0 to i128
  %tobool.i26 = icmp slt i128 %0, 0
  br label %_ZN5boost4math4signIgEEiRKT_.exit30

_ZN5boost4math4signIgEEiRKT_.exit30:              ; preds = %cond.false.i28, %if.then33
  unreachable
}

