; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s

; Make sure code generation does not break in case an 'error block' is detected
; outside of the scope. In this situation, we should not affect code generation.

; CHECK:        polly.cond:
; CHECK-NEXT:   ptrtoint ptr %tmp8 to i64
; CHECK-NEXT:   icmp sle i64
; CHECK-NEXT:   ptrtoint ptr %tmp8 to i64
; CHECK-NEXT:   icmp sge i64
; CHECK-NEXT:   or i1
; CHECK-NEXT:   label %polly.then, label %polly.else

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

declare void @widget()

define void @baz() {
bb:
  br label %bb1

bb1:
  br i1 undef, label %bb5, label %bb2

bb2:
  %tmp = call ptr @pluto()
  br label %bb6

bb5:
  call void @widget()
  br label %bb7

bb6:
  br label %bb7

bb7:
  %tmp8 = phi ptr [ %tmp, %bb6 ], [ null, %bb5 ]
  br label %bb9

bb9:
  %tmp10 = icmp eq ptr %tmp8, null
  br i1 %tmp10, label %bb12, label %bb11

bb11:
  br label %bb12

bb12:
  %tmp13 = phi ptr [ undef, %bb9 ], [ undef, %bb11 ]
  ret void
}

declare ptr @pluto()
