; RUN: opt < %s -dfsan -dfsan-track-select-control-flow=1 -S | FileCheck %s --check-prefixes=CHECK,TRACK_CF,TRACK_CF_LEGACY
; RUN: opt < %s -dfsan -dfsan-track-select-control-flow=0 -S | FileCheck %s --check-prefixes=CHECK,NO_TRACK_CF,NO_TRACK_CF_LEGACY
; RUN: opt < %s -dfsan -dfsan-fast-16-labels -dfsan-track-select-control-flow=1 -S | FileCheck %s --check-prefixes=CHECK,TRACK_CF,TRACK_CF_FAST
; RUN: opt < %s -dfsan -dfsan-fast-16-labels -dfsan-track-select-control-flow=0 -S | FileCheck %s --check-prefixes=CHECK,NO_TRACK_CF,NO_TRACK_CF_FAST
; RUN: opt < %s -dfsan -dfsan-fast-8-labels -dfsan-track-select-control-flow=1 -S | FileCheck %s --check-prefixes=CHECK,TRACK_CF,TRACK_CF_FAST
; RUN: opt < %s -dfsan -dfsan-fast-8-labels -dfsan-track-select-control-flow=0 -S | FileCheck %s --check-prefixes=CHECK,NO_TRACK_CF,NO_TRACK_CF_FAST
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i8 @select8(i1 %c, i8 %t, i8 %f) {
  ; TRACK_CF: @"dfs$select8"
  ; TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+2]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+3]] = select i1 %c, i[[#SBITS]] %[[#R+1]], i[[#SBITS]] %[[#R]]
  ; TRACK_CF_LEGACY: %[[#R+4]] = icmp ne i[[#SBITS]] %[[#R+2]], %[[#R+3]]
  ; TRACK_CF_LEGACY: %[[#R+6]] = call {{.*}} i[[#SBITS]] @__dfsan_union(i[[#SBITS]] {{.*}} %[[#R+2]], i[[#SBITS]] {{.*}} %[[#R+3]])
  ; TRACK_CF_LEGACY: %[[#RO:]] = phi i[[#SBITS]] [ %[[#R+6]], {{.*}} ], [ %[[#R+2]], {{.*}} ]
  ; COMM: The union is simply an OR when fast labels are used.
  ; TRACK_CF_FAST: %[[#RO:]] = or i[[#SBITS]] %[[#R+2]], %[[#R+3]]
  ; TRACK_CF: %a = select i1 %c, i8 %t, i8 %f
  ; TRACK_CF: store i[[#SBITS]] %[[#RO]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: ret i8 %a

  ; NO_TRACK_CF: @"dfs$select8"
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
  ; TRACK_CF: @"dfs$select8e"
  ; TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF_LEGACY: %[[#R+2]] = icmp ne i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; TRACK_CF_LEGACY: %[[#R+4]] = call {{.*}} i[[#SBITS]] @__dfsan_union(i[[#SBITS]] {{.*}} %[[#R+1]], i[[#SBITS]] {{.*}} %[[#R]])
  ; TRACK_CF_LEGACY: %[[#RO:]] = phi i[[#SBITS]] [ %[[#R+4]], {{.*}} ], [ %[[#R+1]], {{.*}} ]
  ; COMM: The union is simply an OR when fast labels are used.
  ; TRACK_CF_FAST: %[[#RO:]] = or i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; TRACK_CF: %a = select i1 %c, i8 %tf, i8 %tf
  ; TRACK_CF: store i[[#SBITS]] %[[#RO]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: ret i8 %a

  ; NO_TRACK_CF: @"dfs$select8e"
  ; NO_TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %a = select i1 %c, i8 %tf, i8 %tf
  ; NO_TRACK_CF: store i[[#SBITS]] %[[#R]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: ret i8 %a

  %a = select i1 %c, i8 %tf, i8 %tf
  ret i8 %a
}

define <4 x i8> @select8v(<4 x i1> %c, <4 x i8> %t, <4 x i8> %f) {
  ; TRACK_CF: @"dfs$select8v"
  ; TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: %[[#R+2]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF_LEGACY: %[[#R+3]] = icmp ne i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; TRACK_CF_LEGACY: %[[#R+5]] = call {{.*}} i[[#SBITS]] @__dfsan_union(i[[#SBITS]] {{.*}} %[[#R+1]], i[[#SBITS]] zeroext %[[#R]])
  ; TRACK_CF_LEGACY: %[[#R+7]] = phi i[[#SBITS]] [ %[[#R+5]], {{.*}} ], [ %[[#R+1]], {{.*}} ]
  ; TRACK_CF_LEGACY: %[[#R+8]] = icmp ne i[[#SBITS]] %[[#R+2]], %[[#R+7]]
  ; TRACK_CF_LEGACY: %[[#R+10]] = call {{.*}} i[[#SBITS]] @__dfsan_union(i[[#SBITS]] {{.*}} %[[#R+2]], i[[#SBITS]] zeroext %[[#R+7]])
  ; TRACK_CF_LEGACY: %[[#RO:]] = phi i[[#SBITS]] [ %[[#R+10]], {{.*}} ], [ %[[#R+2]], {{.*}} ]
  ; COMM: The union is simply an OR when fast labels are used.
  ; TRACK_CF_FAST: %[[#R+3]] = or i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; TRACK_CF_FAST: %[[#RO:]] = or i[[#SBITS]] %[[#R+2]], %[[#R+3]]
  ; TRACK_CF: %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ; TRACK_CF: store i[[#SBITS]] %[[#RO]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; TRACK_CF: ret <4 x i8> %a

  ; NO_TRACK_CF: @"dfs$select8v"
  ; NO_TRACK_CF: %[[#R:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; NO_TRACK_CF: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: %[[#R+2]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF_LEGACY: %[[#R+3]] = icmp ne i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; NO_TRACK_CF_LEGACY: %[[#R+5]] = call {{.*}} i[[#SBITS]] @__dfsan_union(i[[#SBITS]] {{.*}} %[[#R+1]], i[[#SBITS]] {{.*}} %[[#R]])
  ; NO_TRACK_CF_LEGACY: %[[#RO:]] = phi i[[#SBITS]] [ %6, {{.*}} ], [ %2, {{.*}} ]
  ; COMM: The union is simply an OR when fast labels are used.
  ; NO_TRACK_CF_FAST: %[[#RO:]] = or i[[#SBITS]] %[[#R+1]], %[[#R]]
  ; NO_TRACK_CF: %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ; NO_TRACK_CF: store i[[#SBITS]] %[[#RO]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; NO_TRACK_CF: ret <4 x i8> %a

  %a = select <4 x i1> %c, <4 x i8> %t, <4 x i8> %f
  ret <4 x i8> %a
}
