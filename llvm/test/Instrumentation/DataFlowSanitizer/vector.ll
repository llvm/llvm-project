; RUN: opt < %s -dfsan -dfsan-args-abi -S | FileCheck %s --check-prefixes=CHECK,ARGS_ABI
; RUN: opt < %s -dfsan -S | FileCheck %s --check-prefixes=CHECK,TLS_ABI
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define <4 x i4> @pass_vector(<4 x i4> %v) {
  ; ARGS_ABI-LABEL: @pass_vector.dfsan
  ; ARGS_ABI-SAME: (<4 x i4> %[[VEC:.*]], i[[#SBITS]] %[[LABEL:.*]])
  ; ARGS_ABI-NEXT: %[[#REG:]] = insertvalue { <4 x i4>, i[[#SBITS]] } undef, <4 x i4> %[[VEC]], 0
  ; ARGS_ABI-NEXT: %[[#REG+1]] = insertvalue { <4 x i4>, i[[#SBITS]] } %[[#REG]], i[[#SBITS]] %[[LABEL]], 1
  ; ARGS_ABI-NEXT: ret { <4 x i4>, i[[#SBITS]] }

  ; TLS_ABI-LABEL: @pass_vector.dfsan
  ; TLS_ABI-NEXT: %[[#REG:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; TLS_ABI-NEXT: store i[[#SBITS]] %[[#REG]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TLS_ABI-NEXT: ret <4 x i4> %v
  ret <4 x i4> %v
}

define void @load_update_store_vector(<4 x i4>* %p) {
  ; TLS_ABI-LABEL: @load_update_store_vector.dfsan
  ; TLS_ABI: {{.*}} = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2

  %v = load <4 x i4>, <4 x i4>* %p
  %e2 = extractelement <4 x i4> %v, i32 2
  %v1 = insertelement <4 x i4> %v, i4 %e2, i32 0
  store <4 x i4> %v1, <4 x i4>* %p
  ret void
}

define <4 x i1> @icmp_vector(<4 x i8> %a, <4 x i8> %b) {
  ; TLS_ABI-LABEL: @icmp_vector.dfsan
  ; TLS_ABI-NEXT: %[[B:.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; TLS_ABI-NEXT: %[[A:.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TLS_ABI:       %[[L:.*]] = or i[[#SBITS]] %[[A]], %[[B]]

  ; TLS_ABI: %r = icmp eq <4 x i8> %a, %b
  ; TLS_ABI: store i[[#SBITS]] %[[L]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TLS_ABI: ret <4 x i1> %r

  %r = icmp eq <4 x i8> %a, %b
  ret <4 x i1> %r
}

define <2 x i32> @const_vector() {
  ; TLS_ABI-LABEL: @const_vector.dfsan
  ; TLS_ABI-NEXT: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; TLS_ABI-NEXT: ret <2 x i32> <i32 42, i32 11>

  ret <2 x i32> < i32 42, i32 11 >
}

define <4 x i4> @call_vector(<4 x i4> %v) {
  ; TLS_ABI-LABEL: @call_vector.dfsan
  ; TLS_ABI-NEXT: %[[V:.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; TLS_ABI-NEXT: store i[[#SBITS]] %[[V]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TLS_ABI-NEXT: %r = call <4 x i4> @pass_vector.dfsan(<4 x i4> %v)
  ; TLS_ABI-NEXT: %_dfsret = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TLS_ABI-NEXT: store i[[#SBITS]] %_dfsret, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TLS_ABI-NEXT: ret <4 x i4> %r

  %r = call <4 x i4> @pass_vector(<4 x i4> %v)
  ret <4 x i4> %r
}
