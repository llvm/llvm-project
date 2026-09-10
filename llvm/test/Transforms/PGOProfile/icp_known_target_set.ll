; A call whose callee is a phi of functions already dispatches over a known set
; of targets, so it is not promoted. Promoting it there would not devirtualize
; anything the compiler cannot see on its own, and it would keep the phi from
; being folded into a lookup table.

; RUN: opt < %s -passes=pgo-icall-prom -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@target = external global ptr

define i64 @handler_a(ptr %arg) {
  ret i64 0
}

define i64 @handler_b(ptr %arg) {
  ret i64 1
}

define i64 @handler_c(ptr %arg) {
  ret i64 2
}

define i64 @loaded_target(ptr %arg) {
  ret i64 3
}

define i64 @dispatch_over_phi(i8 %op, ptr %arg) {
; CHECK-LABEL: define i64 @dispatch_over_phi(
; CHECK-NOT: icmp eq ptr
; CHECK: %call = call i64 %handler(ptr %arg)
entry:
  switch i8 %op, label %c [
    i8 0, label %a
    i8 1, label %b
  ]

a:
  br label %dispatch

b:
  br label %dispatch

c:
  br label %dispatch

dispatch:
  %handler = phi ptr [ @handler_a, %a ], [ @handler_b, %b ], [ @handler_c, %c ]
  %call = call i64 %handler(ptr %arg), !prof !0
  ret i64 %call
}

; A callee that is not a phi of functions is promoted as before.

define i64 @dispatch_over_load(ptr %arg) {
; CHECK-LABEL: define i64 @dispatch_over_load(
; CHECK: icmp eq ptr %handler, @loaded_target
; CHECK: call i64 @loaded_target(ptr %arg)
entry:
  %handler = load ptr, ptr @target, align 8
  %call = call i64 %handler(ptr %arg), !prof !1
  ret i64 %call
}

; A phi with an unknown incoming value is promoted as before.

define i64 @dispatch_over_mixed_phi(i1 %cond, ptr %arg) {
; CHECK-LABEL: define i64 @dispatch_over_mixed_phi(
; CHECK: icmp eq ptr %handler, @handler_a
; CHECK: call i64 @handler_a(ptr %arg)
entry:
  %loaded = load ptr, ptr @target, align 8
  br i1 %cond, label %a, label %other

a:
  br label %dispatch

other:
  br label %dispatch

dispatch:
  %handler = phi ptr [ @handler_a, %a ], [ %loaded, %other ]
  %call = call i64 %handler(ptr %arg), !prof !2
  ret i64 %call
}

!0 = !{!"VP", i32 0, i64 10000, i64 9552685272606748865, i64 9000}
!1 = !{!"VP", i32 0, i64 10000, i64 1327993795398320997, i64 9000}
!2 = !{!"VP", i32 0, i64 10000, i64 9552685272606748865, i64 9000}
