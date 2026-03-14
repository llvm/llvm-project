; RUN: opt < %s -passes=dfsan -S | FileCheck %s
; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1 -S | FileCheck %s --check-prefixes=CHECK,CHECK_ORIGIN
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @f(i32)

; CHECK-LABEL: @inner_callee.dfsan
define i32 @inner_callee(i32) {
  %r = call i32 @f(i32 %0)

  ; COMM: Store here will be loaded in @outer_caller
  ; CHECK: store{{.*}}__dfsan_retval_tls
  ; CHECK_ORIGIN-NEXT: store{{.*}}__dfsan_retval_origin_tls
  ; CHECK-NEXT: ret i32
  ret i32 %r
}

; CHECK-LABEL: @musttail_call.dfsan
define i32 @musttail_call(i32) {
  ; CHECK: store{{.*}}__dfsan_arg_tls
  ; CHECK-NEXT: musttail call i32 @inner_callee.dfsan
  %r = musttail call i32 @inner_callee(i32 %0)

  ; For "musttail" calls we can not insert any shadow manipulating code between
  ; call and the return instruction. And we don't need to, because everything is
  ; taken care of in the callee.
  ; This is similar to the function above, but the last load and store of
  ; __dfsan_retval_tls can be elided because we know about the musttail.

  ; CHECK-NEXT: ret i32
  ret i32 %r
}

; CHECK-LABEL: @outer_caller.dfsan
define i32 @outer_caller() {
  ; CHECK: call{{.*}}@musttail_call.dfsan
  ; CHECK-NEXT: load{{.*}}__dfsan_retval_tls
  ; CHECK_ORIGIN-NEXT: load{{.*}}__dfsan_retval_origin_tls
  %r = call i32 @musttail_call(i32 0)

  ; CHECK-NEXT: store{{.*}}__dfsan_retval_tls
  ; CHECK_ORIGIN-NEXT: store{{.*}}__dfsan_retval_origin_tls
  ; CHECK-NEXT: ret i32
  ret i32 %r
}

declare ptr @mismatching_callee(i32)

; CHECK-LABEL: define ptr @mismatching_musttail_call.dfsan
define ptr @mismatching_musttail_call(i32) {
  %r = musttail call ptr @mismatching_callee(i32 %0)
  ; CHECK: musttail call ptr @mismatching_callee.dfsan
  ; COMM: No instrumentation between call and ret.
  ; CHECK-NEXT: ret ptr
  ret ptr %r
}
