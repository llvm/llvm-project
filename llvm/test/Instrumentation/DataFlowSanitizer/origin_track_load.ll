; RUN: opt < %s -dfsan -dfsan-track-origins=2 -dfsan-fast-8-labels -S | FileCheck %s
; RUN: opt < %s -dfsan -dfsan-track-origins=2 -dfsan-fast-16-labels -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i64 @load64(i64* %p) {
  ; CHECK-LABEL: @"dfs$load64"

  ; CHECK-NEXT: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK-NEXT: %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]

  ; CHECK-NEXT: %[[#INTP:]] = bitcast i64* %p to i8*
  ; CHECK-NEXT: %[[#LABEL_ORIGIN:]] = call zeroext i64 @__dfsan_load_label_and_origin(i8* %[[#INTP]], i64 8)
  ; CHECK-NEXT: %[[#LABEL_ORIGIN_H32:]] = lshr i64 %[[#LABEL_ORIGIN]], 32
  ; CHECK-NEXT: %[[#LABEL:]] = trunc i64 %[[#LABEL_ORIGIN_H32]] to i[[#SBITS]]
  ; CHECK-NEXT: %[[#ORIGIN:]] = trunc i64 %[[#LABEL_ORIGIN]] to i32
  ; CHECK-NEXT: %[[#ORIGIN_CHAINED:]] = call zeroext i32 @__dfsan_chain_origin_if_tainted(i[[#SBITS]] zeroext %[[#LABEL]], i32 zeroext %[[#ORIGIN]])

  ; CHECK-NEXT: %[[#LABEL:]] = or i[[#SBITS]] %[[#LABEL]], %[[#PS]]
  ; CHECK-NEXT: %[[#NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; CHECK-NEXT: %[[#ORIGIN_SEL:]] = select i1 %[[#NZ]], i32 %[[#PO]], i32 %[[#ORIGIN_CHAINED]]

  ; CHECK-NEXT: %a = load i64, i64* %p
  ; CHECK-NEXT: store i[[#SBITS]] %[[#LABEL]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT: store i32 %[[#ORIGIN_SEL]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i64, i64* %p
  ret i64 %a
}
