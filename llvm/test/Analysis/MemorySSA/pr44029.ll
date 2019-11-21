; RUN: opt -loop-simplifycfg -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @eggs(...)

declare void @spam()

; CHECK-LABEL: @f()
define void @f() personality i8* bitcast (i32 (...)* @eggs to i8*) {
bb:
  invoke void @spam()
          to label %bb2 unwind label %bb4

bb2:                                              ; preds = %bb
  invoke void @spam()
          to label %bb8 unwind label %bb5

bb4:                                              ; preds = %bb
  %tmp = landingpad { i8*, i32 }
          cleanup
  resume { i8*, i32 } undef

bb5:                                              ; preds = %bb2
  %tmp6 = landingpad { i8*, i32 }
          cleanup
  unreachable

bb8:                                              ; preds = %bb13, %bb2
  br label %bb10

bb10:                                             ; preds = %bb8
  invoke void @spam()
          to label %bb11 unwind label %bb20

bb11:                                             ; preds = %bb10
  invoke void @spam()
          to label %bb12 unwind label %bb22

bb12:                                             ; preds = %bb11
  invoke void @spam()
          to label %bb13 unwind label %bb24

bb13:                                             ; preds = %bb12
  br label %bb8

bb20:                                             ; preds = %bb10
  %tmp21 = landingpad { i8*, i32 }
          cleanup
  unreachable

bb22:                                             ; preds = %bb11
  %tmp23 = landingpad { i8*, i32 }
          cleanup
  unreachable

bb24:                                             ; preds = %bb12
  %tmp25 = landingpad { i8*, i32 }
          cleanup
  unreachable
}
