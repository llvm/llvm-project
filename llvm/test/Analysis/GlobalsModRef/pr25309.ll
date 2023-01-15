; RUN: opt -aa-pipeline=basic-aa,globals-aa -passes='require<globals-aa>,gvn' < %s -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; @o and @m are initialized to @i, so they should not be classed as
; indirect globals referring only to allocation functions.
@o = internal global ptr @i, align 8
@m = internal global ptr @i, align 8
@i = internal global i32 0, align 4

; CHECK-LABEL: @f
define i1 @f() {
entry:
  %0 = load ptr, ptr @o, align 8
  store i32 0, ptr %0, align 4
  %1 = load volatile ptr, ptr @m, align 8
  store i32 1, ptr %1, align 4
  ; CHECK: %[[a:.*]] = load ptr
  %2 = load ptr, ptr @o, align 8
  ; CHECK: %[[b:.*]] = load i32, ptr %[[a]]
  %3 = load i32, ptr %2, align 4
  ; CHECK: %[[c:.*]] = icmp ne i32 %[[b]], 0
  %tobool.i = icmp ne i32 %3, 0
  ; CHECK: ret i1 %[[c]]
  ret i1 %tobool.i
}
