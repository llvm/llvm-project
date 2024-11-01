; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-load=true -S | FileCheck %s --check-prefixes=CHECK,COMBINE_LOAD_PTR
; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]


define {} @load0(ptr %p) {
  ; CHECK-LABEL:           @load0.dfsan
  ; CHECK-NEXT:            %a = load {}, ptr %p, align 1
  ; CHECK-NEXT:            store {} zeroinitializer, ptr @__dfsan_retval_tls, align [[ALIGN:2]]
  ; CHECK-NEXT:            ret {} %a

  %a = load {}, ptr %p
  ret {} %a
}

define i8 @load8(ptr %p) {
  ; CHECK-LABEL:           @load8.dfsan
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#SHADOW:]] = load i[[#SBITS]], ptr %[[#SHADOW_PTR]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; CHECK-NEXT:            %a = load i8, ptr %p
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i8 %a

  %a = load i8, ptr %p
  ret i8 %a
}

define i16 @load16(ptr %p) {
  ; CHECK-LABEL:           @load16.dfsan
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#SHADOW_PTR+1]] = getelementptr i[[#SBITS]], ptr %[[#SHADOW_PTR]], i64 1
  ; CHECK-NEXT:            %[[#SHADOW:]]  = load i[[#SBITS]], ptr %[[#SHADOW_PTR]]
  ; CHECK-NEXT:            %[[#SHADOW+1]] = load i[[#SBITS]], ptr %[[#SHADOW_PTR+1]]

  ; CHECK-NEXT:            %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#SHADOW+1]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; CHECK-NEXT:            %a = load i16, ptr %p
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i16 %a

  %a = load i16, ptr %p
  ret i16 %a
}

define i32 @load32(ptr %p) {
  ; CHECK-LABEL:           @load32.dfsan
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = load i[[#WSBITS:mul(SBITS,4)]], ptr %[[#SHADOW_PTR]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i[[#WSBITS]] %[[#WIDE_SHADOW]], [[#mul(SBITS,2)]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i[[#WSBITS]] %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i[[#WSBITS]] %[[#WIDE_SHADOW]], [[#SBITS]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i[[#WSBITS]] %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#SHADOW:]] = trunc i[[#WSBITS]] %[[#WIDE_SHADOW]] to i[[#SBITS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; CHECK-NEXT:            %a = load i32, ptr %p, align 4
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i32 %a

  %a = load i32, ptr %p
  ret i32 %a
}

define i64 @load64(ptr %p) {
  ; CHECK-LABEL:           @load64.dfsan
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = load i64, ptr %[[#SHADOW_PTR]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 32
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 16
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#SHADOW:]] = trunc i64 %[[#WIDE_SHADOW]] to i[[#SBITS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; CHECK-NEXT:            %a = load i64, ptr %p, align 8
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i64 %a

  %a = load i64, ptr %p
  ret i64 %a
}

define i128 @load128(ptr %p) {
  ; CHECK-LABEL:           @load128.dfsan
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = load i64, ptr %[[#SHADOW_PTR]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_PTR2:]] = getelementptr i64, ptr %[[#SHADOW_PTR]], i64 1
  ; CHECK-NEXT:            %[[#WIDE_SHADOW2:]] = load i64, ptr %[[#WIDE_SHADOW_PTR2]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW2]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 32
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 16
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#SHADOW:]] = trunc i64 %[[#WIDE_SHADOW]] to i[[#SBITS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; CHECK-NEXT:            %a = load i128, ptr %p, align 8
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i128 %a

  %a = load i128, ptr %p
  ret i128 %a
}


define i17 @load17(ptr %p) {
  ; CHECK-LABEL:           @load17.dfsan
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#SHADOW:]] = call zeroext i8 @__dfsan_union_load(ptr %[[#SHADOW_PTR]], i64 3)
  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; CHECK-NEXT:            %a = load i17, ptr %p
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i17 %a

  %a = load i17, ptr %p
  ret i17 %a
}

@X = constant i1 1
define i1 @load_global() {
  ; CHECK-LABEL:           @load_global.dfsan
  ; CHECK-NEXT:            %a = load i1, ptr @X
  ; CHECK-NEXT:            store i[[#SBITS]] 0, ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            ret i1 %a

  %a = load i1, ptr @X
  ret i1 %a
}
