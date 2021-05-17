; Test that -dfsan-fast-16-labels mode uses inline ORs rather than calling
; __dfsan_union or __dfsan_union_load.
; RUN: opt < %s -dfsan -dfsan-fast-16-labels -S | FileCheck %s --implicit-check-not="call{{.*}}__dfsan_union" --check-prefixes=CHECK,CHECK16
; RUN: opt < %s -dfsan -dfsan-fast-8-labels -S | FileCheck %s --implicit-check-not="call{{.*}}__dfsan_union" --check-prefixes=CHECK,CHECK8
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i8 @add(i8 %a, i8 %b) {
  ; CHECK-LABEL: define i8 @"dfs$add"
  ; CHECK-DAG: %[[ALABEL:.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; CHECK-DAG: %[[BLABEL:.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK: %[[ADDLABEL:.*]] = or i[[#SBITS]] %[[ALABEL]], %[[BLABEL]]
  ; CHECK: %c = add i8 %a, %b
  ; CHECK: store i[[#SBITS]] %[[ADDLABEL]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK: ret i8 %c
  %c = add i8 %a, %b
  ret i8 %c
}

define i8 @load8(i8* %p) {
  ; CHECK-LABEL:  define i8 @"dfs$load8"
  ; CHECK-SAME:   (i8* %[[PADDR:.*]])
  ; CHECK-NEXT:   %[[#ARG:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   %[[#R:]] = ptrtoint i8* %[[PADDR]] to i64
  ; CHECK-NEXT:   %[[#PS:R+1]] = and i64 %[[#R]], [[#%.10d,MASK:]]
  ; CHECK16-NEXT: %[[#PS:R+2]] = mul i64 %[[#R+1]], 2
  ; CHECK-NEXT:   %[[#SADDR:]] = inttoptr i64 %[[#PS]] to i[[#SBITS]]*
  ; CHECK-NEXT:   %[[#S:]] = load i[[#SBITS]], i[[#SBITS]]* %[[#SADDR]]
  ; CHECK-NEXT:   %[[#S_OUT:S+1]] = or i[[#SBITS]] %[[#S]], %[[#ARG]]
  ; CHECK-NEXT:   %a = load i8, i8* %p
  ; CHECK-NEXT:   store i[[#SBITS]] %[[#S_OUT]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   ret i8 %a

  %a = load i8, i8* %p
  ret i8 %a
}

define i16 @load16(i16* %p) {
  ; CHECK-LABEL:  define i16 @"dfs$load16"
  ; CHECK-SAME:   (i16* %[[PADDR:.*]])
  ; CHECK-NEXT:   %[[#ARG:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   %[[#R:]] = ptrtoint i16* %[[PADDR]] to i64
  ; CHECK-NEXT:   %[[#PS:R+1]] = and i64 %[[#R]], [[#%.10d,MASK:]]
  ; CHECK16-NEXT: %[[#PS:R+2]] = mul i64 %[[#R+1]], 2
  ; CHECK-NEXT:   %[[#SADDR:]]  = inttoptr i64 %[[#PS]] to i[[#SBITS]]*
  ; CHECK-NEXT:   %[[#SADDR+1]] = getelementptr i[[#SBITS]], i[[#SBITS]]* %[[#SADDR]], i64 1
  ; CHECK-NEXT:   %[[#S:]]  = load i[[#SBITS]], i[[#SBITS]]* %[[#SADDR]]
  ; CHECK-NEXT:   %[[#S+1]] = load i[[#SBITS]], i[[#SBITS]]* %[[#SADDR+1]]
  ; CHECK-NEXT:   %[[#S+2]] = or i[[#SBITS]] %[[#S]], %[[#S+1]]
  ; CHECK-NEXT:   %[[#S_OUT:S+3]] = or i[[#SBITS]] %[[#S+2]], %[[#ARG]]
  ; CHECK-NEXT:   %a = load i16, i16* %p
  ; CHECK-NEXT:   store i[[#SBITS]] %[[#S_OUT]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   ret i16 %a

  %a = load i16, i16* %p
  ret i16 %a
}

define i32 @load32(i32* %p) {
  ; CHECK-LABEL: define i32 @"dfs$load32"
  ; CHECK-SAME:   (i32* %[[PADDR:.*]])
  ; CHECK-NEXT:   %[[#ARG:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   %[[#R:]] = ptrtoint i32* %[[PADDR]] to i64
  ; CHECK-NEXT:   %[[#PS:R+1]] = and i64 %[[#R]], [[#%.10d,MASK:]]
  ; CHECK16-NEXT: %[[#PS:R+2]] = mul i64 %[[#R+1]], 2
  ; CHECK-NEXT:   %[[#SADDR:]] = inttoptr i64 %[[#PS]] to i[[#SBITS]]*
  ; CHECK-NEXT:   %[[#SADDR+1]] = bitcast i[[#SBITS]]* %[[#SADDR]] to i[[#WBITS:mul(SBITS,4)]]*
  ; CHECK-NEXT:   %[[#WS:]]  = load i[[#WBITS]], i[[#WBITS]]* %[[#SADDR+1]]
  ; CHECK-NEXT:   %[[#WS+1]] = lshr i[[#WBITS]] %[[#WS]], [[#mul(SBITS,2)]]
  ; CHECK-NEXT:   %[[#WS+2]] = or i[[#WBITS]] %[[#WS]], %[[#WS+1]]
  ; CHECK-NEXT:   %[[#WS+3]] = lshr i[[#WBITS]] %[[#WS+2]], [[#SBITS]]
  ; CHECK-NEXT:   %[[#WS+4]] = or i[[#WBITS]] %[[#WS+2]], %[[#WS+3]]
  ; CHECK-NEXT:   %[[#WS+5]] = trunc i[[#WBITS]] %[[#WS+4]] to i[[#SBITS]]
  ; CHECK-NEXT:   %[[#S_OUT:WS+6]] = or i[[#SBITS]] %[[#WS+5]], %[[#ARG]]
  ; CHECK-NEXT:   %a = load i32, i32* %p
  ; CHECK-NEXT:   store i[[#SBITS]] %[[#S_OUT]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   ret i32 %a

  %a = load i32, i32* %p
  ret i32 %a
}

define i64 @load64(i64* %p) {
  ; CHECK-LABEL:  define i64 @"dfs$load64"
  ; CHECK-SAME:   (i64* %[[PADDR:.*]])
  ; CHECK-NEXT:   %[[#ARG:]]        = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   %[[#R:]]          = ptrtoint i64* %[[PADDR]] to i64
  ; CHECK-NEXT:   %[[#PS:R+1]]      = and i64 %[[#R]], [[#%.10d,MASK:]]
  ; CHECK16-NEXT: %[[#PS:R+2]]      = mul i64 %[[#R+1]], 2
  ; CHECK-NEXT:   %[[#SADDR:]]      = inttoptr i64 %[[#PS]] to i[[#SBITS]]*
  ; CHECK-NEXT:   %[[#SADDR+1]]     = bitcast i[[#SBITS]]* %[[#SADDR]] to i64*
  ; CHECK-NEXT:   %[[#WS:]]         = load i64, i64* %[[#SADDR+1]]

  ; COMM: On fast16, the 2x64 shadow bits need to be ORed first.
  ; CHECK16-NEXT: %[[#SADDR_NEXT:]] = getelementptr i64, i64* %[[#SADDR+1]], i64 1
  ; CHECK16-NEXT: %[[#WS_NEXT:]]    = load i64, i64* %[[#SADDR_NEXT]]
  ; CHECK16-NEXT: %[[#WS:]]         = or i64 %[[#WS]], %[[#WS_NEXT]]
  ; CHECK16-NEXT: %[[#WS+1]]        = lshr i64 %[[#WS]], 32
  ; CHECK16-NEXT: %[[#WS+2]]        = or i64 %[[#WS]], %[[#WS+1]]
  ; CHECK16-NEXT: %[[#WS+3]]        = lshr i64 %[[#WS+2]], 16
  ; CHECK16-NEXT: %[[#WS+4]]        = or i64 %[[#WS+2]], %[[#WS+3]]
  ; CHECK16-NEXT: %[[#WS+5]]        = trunc i64 %[[#WS+4]] to i[[#SBITS]]
  ; CHECK16-NEXT: %[[#S_OUT:]]      = or i[[#SBITS]] %[[#WS+5]], %[[#ARG]]

  ; COMM: On fast8, no need to OR the wide shadow but one more shift is needed.
  ; CHECK8-NEXT: %[[#WS+1]]         = lshr i64 %[[#WS]], 32
  ; CHECK8-NEXT: %[[#WS+2]]         = or i64 %[[#WS]], %[[#WS+1]]
  ; CHECK8-NEXT: %[[#WS+3]]         = lshr i64 %[[#WS+2]], 16
  ; CHECK8-NEXT: %[[#WS+4]]         = or i64 %[[#WS+2]], %[[#WS+3]]
  ; CHECK8-NEXT: %[[#WS+5]]         = lshr i64 %[[#WS+4]], 8
  ; CHECK8-NEXT: %[[#WS+6]]         = or i64 %[[#WS+4]], %[[#WS+5]]
  ; CHECK8-NEXT: %[[#WS+7]]         = trunc i64 %[[#WS+6]] to i[[#SBITS]]
  ; CHECK8-NEXT: %[[#S_OUT:]]       = or i[[#SBITS]] %[[#WS+7]], %[[#ARG]]

  ; CHECK-NEXT:   %a = load i64, i64* %p
  ; CHECK-NEXT:   store i[[#SBITS]] %[[#S_OUT]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   ret i64 %a

  %a = load i64, i64* %p
  ret i64 %a
}

define i128 @load128(i128* %p) {
  ; CHECK-LABEL:  define i128 @"dfs$load128"
  ; CHECK-SAME:   (i128* %[[PADDR:.*]])
  ; CHECK-NEXT:   %[[#ARG:]]    = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT:   %[[#R:]]      = ptrtoint i128* %[[PADDR]] to i64
  ; CHECK-NEXT:   %[[#PS:R+1]]  = and i64 %[[#R]], [[#%.10d,MASK:]]
  ; CHECK16-NEXT: %[[#PS:R+2]]  = mul i64 %[[#R+1]], 2
  ; CHECK-NEXT:   %[[#SADDR:]]  = inttoptr i64 %[[#PS]] to i[[#SBITS]]*
  ; CHECK-NEXT:   %[[#SADDR+1]] = bitcast i[[#SBITS]]* %[[#SADDR]] to i64*
  ; CHECK-NEXT:   %[[#S:]]      = load i64, i64* %[[#SADDR+1]]
  ; CHECK-NEXT:   %[[#S+1]]     = getelementptr i64, i64* %[[#SADDR+1]], i64 1
  ; CHECK-NEXT:   %[[#S+2]]     = load i64, i64* %[[#S+1]]
  ; CHECK-NEXT:   %[[#WS:S+3]]  = or i64 %[[#S]], %[[#S+2]]

  ; COMM: On fast16, we need to OR 4x64bits for the wide shadow, before ORing its bytes.
  ; CHECK16-NEXT: %[[#S+4]]     = getelementptr i64, i64* %[[#S+1]], i64 1
  ; CHECK16-NEXT: %[[#S+5]]     = load i64, i64* %[[#S+4]]
  ; CHECK16-NEXT: %[[#S+6]]     = or i64 %[[#S+3]], %[[#S+5]]
  ; CHECK16-NEXT: %[[#S+7]]     = getelementptr i64, i64* %[[#S+4]], i64 1
  ; CHECK16-NEXT: %[[#S+8]]     = load i64, i64* %[[#S+7]]
  ; CHECK16-NEXT: %[[#WS:S+9]]  = or i64 %[[#S+6]], %[[#S+8]]
  ; CHECK16-NEXT: %[[#WS+1]]    = lshr i64 %[[#WS]], 32
  ; CHECK16-NEXT: %[[#WS+2]]    = or i64 %[[#WS]], %[[#WS+1]]
  ; CHECK16-NEXT: %[[#WS+3]]    = lshr i64 %[[#WS+2]], 16
  ; CHECK16-NEXT: %[[#WS+4]]    = or i64 %[[#WS+2]], %[[#WS+3]]
  ; CHECK16-NEXT: %[[#WS+5]]    = trunc i64 %[[#WS+4]] to i[[#SBITS]]
  ; CHECK16-NEXT: %[[#S_OUT:]]  = or i[[#SBITS]] %[[#WS+5]], %[[#ARG]]

  ; COMM: On fast8, we need to OR 2x64bits for the wide shadow, before ORing its bytes (one more shift).
  ; CHECK8-NEXT: %[[#WS+1]]     = lshr i64 %[[#WS]], 32
  ; CHECK8-NEXT: %[[#WS+2]]     = or i64 %[[#WS]], %[[#WS+1]]
  ; CHECK8-NEXT: %[[#WS+3]]     = lshr i64 %[[#WS+2]], 16
  ; CHECK8-NEXT: %[[#WS+4]]     = or i64 %[[#WS+2]], %[[#WS+3]]
  ; CHECK8-NEXT: %[[#WS+5]]     = lshr i64 %[[#WS+4]], 8
  ; CHECK8-NEXT: %[[#WS+6]]     = or i64 %[[#WS+4]], %[[#WS+5]]
  ; CHECK8-NEXT: %[[#WS+7]]     = trunc i64 %[[#WS+6]] to i[[#SBITS]]
  ; CHECK8-NEXT: %[[#S_OUT:]]   = or i[[#SBITS]] %[[#WS+7]], %[[#ARG]]

  ; CHECK-NEXT: %a = load i128, i128* %p
  ; CHECK-NEXT: store i[[#SBITS]] %[[#S_OUT]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK-NEXT: ret i128 %a

  %a = load i128, i128* %p
  ret i128 %a
}
