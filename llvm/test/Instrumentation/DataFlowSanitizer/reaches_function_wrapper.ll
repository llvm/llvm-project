; RUN: opt < %s -passes=dfsan -dfsan-reaches-function-callbacks=1 -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
; RUN: opt < %s -passes=dfsan -dfsan-reaches-function-callbacks=1 -dfsan-abilist=%S/Inputs/abilist.txt -dfsan-track-origins=1 -S | FileCheck %s --check-prefix=ORIGIN
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; A reaches function callback that is uninstrumented according to the ABI list
; and registered with the runtime is called through its wrapper from
; uninstrumented code, so its argument shadows in __dfsan_arg_tls are stale.
; Reporting them would re-enter the callback and recurse infinitely, so the
; wrapper must not get argument callbacks.

declare void @dfsan_set_reaches_function_callback(ptr)

define void @discard_callback(i8 zeroext %label, ptr %file, i32 %line, ptr %function, i32 %extra) {
  ret void
}

define void @register() {
  call void @dfsan_set_reaches_function_callback(ptr @discard_callback)
  ret void
}

; CHECK-LABEL: define linkonce_odr void @"dfsw$discard_callback"
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @discard_callback(
; CHECK-NEXT:    ret void

; ORIGIN-LABEL: define linkonce_odr void @"dfso$discard_callback"
; ORIGIN-NEXT:  entry:
; ORIGIN-NEXT:    call void @discard_callback(
; ORIGIN-NEXT:    ret void
