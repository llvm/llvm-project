; RUN: opt -passes=gvn-hoist -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @wobble(...)

declare void @spam() align 2

; CHECK-LABEL: @f()
define void @f() personality ptr @wobble {
bb:
  %tmp = alloca ptr, align 8
  invoke void @spam()
          to label %bb16 unwind label %bb23

bb16:                                             ; preds = %bb
  %tmp17 = load ptr, ptr %tmp, align 8
  %tmp18 = load ptr, ptr %tmp, align 8
  unreachable

bb23:                                             ; preds = %bb
  %tmp24 = landingpad { ptr, i32 }
          cleanup
  unreachable
}
