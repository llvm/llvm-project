; RUN: opt < %s -dfsan -dfsan-track-select-control-flow=true -S | FileCheck %s --check-prefixes=CHECK,TRACK_CF
; RUN: opt < %s -dfsan -dfsan-track-select-control-flow=false -S | FileCheck %s --check-prefixes=CHECK,NO_TRACK_CF
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i8 @select8(i1 %c, i8 %t, i8 %f) {
  ; TRACK_CF: @select8.dfsan
  ; TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+2]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+3]] = select i1 %c, i[[#SBITS]] %[[#R+1]], i[[#SBITS]] %[[#R]]
  ; TRACK_CF: %[[#RO:]] = or i[[#SBITS]] %[[#R+2]], %[[#R+3]]
  ; TRACK_CF: %a = select i1 %c, i8 %t, i8 %f
  ; TRACK_CF: store i[[#SBITS]] %[[#RO]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: ret i8 %a

  ; NO_TRACK_CF: @select8.dfsan
  ; NO_TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; NO_TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %[[#R+2]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %[[#R+3]] = select i1 %c, i[[#SBITS]] %[[#R+1]], i[[#SBITS]] %[[#R]]
  ; NO_TRACK_CF: %a = select i1 %c, i8 %t, i8 %f
  ; NO_TRACK_CF: store i[[#SBITS]] %[[#R+3]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: ret i8 %a

  %a = select i1 %c, i8 %t, i8 %f
  ret i8 %a
}

define i8 @select8e(i1 %c, i8 %tf) {
  ; TRACK_CF: @select8e.dfsan
  ; TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#RO:]] = or i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; TRACK_CF: %a = select i1 %c, i8 %tf, i8 %tf
  ; TRACK_CF: store i[[#SBITS]] %[[#RO]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: ret i8 %a

  ; NO_TRACK_CF: @select8e.dfsan
  ; NO_TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %a = select i1 %c, i8 %tf, i8 %tf
  ; NO_TRACK_CF: store i[[#SBITS]] %[[#R]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: ret i8 %a

  %a = select i1 %c, i8 %tf, i8 %tf
  ret i8 %a
}

define <4 x i8> @select8v(<4 x i1> %c, <4 x i8> %t, <4 x i8> %f) {
  ; TRACK_CF: @select8v.dfsan
  ; TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+2]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+3]] = or i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; TRACK_CF: %[[#RO:]] = or i[[#SBITS]] %[[#R+2]], %[[#R+3]]
  ; TRACK_CF: %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ; TRACK_CF: store i[[#SBITS]] %[[#RO]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: ret <4 x i8> %a

  ; NO_TRACK_CF: @select8v.dfsan
  ; NO_TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; NO_TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %[[#R+2]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %[[#RO:]] = or i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; NO_TRACK_CF: %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ; NO_TRACK_CF: store i[[#SBITS]] %[[#RO]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: ret <4 x i8> %a

  %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ret <4 x i8> %a
}
