; RUN: opt < %s -dfsan -dfsan-event-callbacks=1 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i8 @load8(i8* %p) {
  ; CHECK: call void @__dfsan_load_callback(i[[#SBITS]] %[[LABEL:.*]], i8* %p)
  ; CHECK: %a = load i8, i8* %p
  ; CHECK: store i[[#SBITS]] %[[LABEL]], i[[#SBITS]]* bitcast ({{.*}}* @__dfsan_retval_tls to i[[#SBITS]]*)

  %a = load i8, i8* %p
  ret i8 %a
}

define void @store8(i8* %p, i8 %a) {
  ; CHECK: store i[[#SBITS]] %[[LABEL:.*]], i[[#SBITS]]* %{{.*}}
  ; CHECK: call void @__dfsan_store_callback(i[[#SBITS]] %[[LABEL]], i8* %p)
  ; CHECK: store i8 %a, i8* %p

  store i8 %a, i8* %p
  ret void
}

define i1 @cmp(i8 %a, i8 %b) {
  ; CHECK: call void @__dfsan_cmp_callback(i[[#SBITS]] %[[CMPLABEL:.*]])
  ; CHECK: %c = icmp ne i8 %a, %b
  ; CHECK: store i[[#SBITS]] %[[CMPLABEL]], i[[#SBITS]]* bitcast ({{.*}}* @__dfsan_retval_tls to i[[#SBITS]]*)

  %c = icmp ne i8 %a, %b
  ret i1 %c
}