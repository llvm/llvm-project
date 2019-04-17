; RUN: opt -jump-threading -S < %s | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare void @helper()
declare i32 @__gxx_personality_v0(...)


define void @pr27840(i8* %call, i1 %A) personality i32(...)* @__gxx_personality_v0 {
entry:
  invoke void @helper()
          to label %invoke.cont unwind label %lpad

; Don't jump threading; we can't split the critical edge from entry to lpad.
; CHECK-LABEL: @pr27840
; CHECK: invoke
; CHECK-NEXT: to label %invoke.cont unwind label %lpad

invoke.cont:
  invoke void @helper()
          to label %nowhere unwind label %lpad

lpad:
  %b = phi i1 [ true, %invoke.cont ], [ false, %entry ]
  landingpad { i8*, i32 }
          cleanup
  %xor = xor i1 %b, %A
  br i1 %xor, label %nowhere, label %invoke.cont

nowhere:
  unreachable
}
