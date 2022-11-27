; RUN: opt < %s -passes=dfsan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define <4 x i4> @pass_vector(<4 x i4> %v) {
  ; CHECK-LABEL: @pass_vector.dfsan
  ; CHECK-NEXT: %[[#REG:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; CHECK-NEXT: store i[[#SBITS]] %[[#REG]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT: ret <4 x i4> %v
  ret <4 x i4> %v
}

define void @load_update_store_vector(ptr %p) {
  ; CHECK-LABEL: @load_update_store_vector.dfsan
  ; CHECK: {{.*}} = load i[[#SBITS]], ptr @__dfsan_arg_tls, align 2

  %v = load <4 x i4>, ptr %p
  %e2 = extractelement <4 x i4> %v, i32 2
  %v1 = insertelement <4 x i4> %v, i4 %e2, i32 0
  store <4 x i4> %v1, ptr %p
  ret void
}

define <4 x i1> @icmp_vector(<4 x i8> %a, <4 x i8> %b) {
  ; CHECK-LABEL: @icmp_vector.dfsan
  ; CHECK-NEXT: %[[B:.*]] = load i[[#SBITS]], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN:2]]
  ; CHECK-NEXT: %[[A:.*]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK:       %[[L:.*]] = or i[[#SBITS]] %[[A]], %[[B]]

  ; CHECK: %r = icmp eq <4 x i8> %a, %b
  ; CHECK: store i[[#SBITS]] %[[L]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK: ret <4 x i1> %r

  %r = icmp eq <4 x i8> %a, %b
  ret <4 x i1> %r
}

define <2 x i32> @const_vector() {
  ; CHECK-LABEL: @const_vector.dfsan
  ; CHECK-NEXT: store i[[#SBITS]] 0, ptr @__dfsan_retval_tls, align 2
  ; CHECK-NEXT: ret <2 x i32> <i32 42, i32 11>

  ret <2 x i32> < i32 42, i32 11 >
}

define <4 x i4> @call_vector(<4 x i4> %v) {
  ; CHECK-LABEL: @call_vector.dfsan
  ; CHECK-NEXT: %[[V:.*]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; CHECK-NEXT: store i[[#SBITS]] %[[V]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK-NEXT: %r = call <4 x i4> @pass_vector.dfsan(<4 x i4> %v)
  ; CHECK-NEXT: %_dfsret = load i[[#SBITS]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT: store i[[#SBITS]] %_dfsret, ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT: ret <4 x i4> %r

  %r = call <4 x i4> @pass_vector(<4 x i4> %v)
  ret <4 x i4> %r
}
