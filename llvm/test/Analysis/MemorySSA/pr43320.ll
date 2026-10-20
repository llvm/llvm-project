; RUN: opt -passes=licm -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: @e(i1 %arg)
define void @e(i1 %arg) {
entry:
  br label %g

g:                                                ; preds = %cleanup, %entry
  %0 = load i32, ptr null, align 4
  %and = and i32 %0, undef
  store i32 %and, ptr null, align 4
  br i1 %arg, label %if.end8, label %if.then

if.then:                                          ; preds = %g
  br i1 %arg, label %k, label %cleanup

k:                                                ; preds = %if.end8, %if.then
  br i1 %arg, label %if.end8, label %cleanup

if.end8:                                          ; preds = %k, %g
  br i1 %arg, label %for.cond.preheader, label %k

for.cond.preheader:                               ; preds = %if.end8
  unreachable

cleanup:                                          ; preds = %k, %if.then
  br label %g
}

