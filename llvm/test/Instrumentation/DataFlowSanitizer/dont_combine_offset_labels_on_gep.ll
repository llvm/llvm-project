; RUN: opt < %s -passes=dfsan -dfsan-combine-offset-labels-on-gep=false -S | FileCheck %s
; RUN: opt < %s -passes=dfsan -dfsan-combine-offset-labels-on-gep=false -dfsan-track-origins=1 -S | FileCheck %s --check-prefixes=CHECK,CHECK_ORIGIN
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]

define i32* @gepop([10 x [20 x i32]]* %p, i32 %a, i32 %b, i32 %c) {
  ; CHECK: @gepop.dfsan
  ; CHECK_ORIGIN: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align [[ALIGN_O:4]]
  ; CHECK: %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN_S:2]]
  ; CHECK: %e = getelementptr [10 x [20 x i32]], [10 x [20 x i32]]* %p, i32 %a, i32 %b, i32 %c
  ; CHECK: store i[[#SBITS]] %[[#PS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN_S]]
  ; CHECK_ORIGIN: store i32 %[[#PO]], i32* @__dfsan_retval_origin_tls, align [[ALIGN_O]]

  %e = getelementptr [10 x [20 x i32]], [10 x [20 x i32]]* %p, i32 %a, i32 %b, i32 %c
  ret i32* %e
}

