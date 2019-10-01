; RUN: opt -licm -enable-mssa-loop-dependency -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-none-eabi"

; CHECK-LABEL: @e()
define void @e() {
entry:
  br label %g

g:                                                ; preds = %cleanup, %entry
  %0 = load i32, i32* null, align 4
  %and = and i32 %0, undef
  store i32 %and, i32* null, align 4
  br i1 undef, label %if.end8, label %if.then

if.then:                                          ; preds = %g
  br i1 undef, label %k, label %cleanup

k:                                                ; preds = %if.end8, %if.then
  br i1 undef, label %if.end8, label %cleanup

if.end8:                                          ; preds = %k, %g
  br i1 undef, label %for.cond.preheader, label %k

for.cond.preheader:                               ; preds = %if.end8
  unreachable

cleanup:                                          ; preds = %k, %if.then
  br label %g
}

