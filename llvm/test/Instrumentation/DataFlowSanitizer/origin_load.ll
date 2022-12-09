; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1 -S | FileCheck %s --check-prefixes=CHECK,COMBINE_LOAD_PTR
; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1 -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define {} @load0(ptr %p) {
  ; CHECK-LABEL: @load0.dfsan
  ; CHECK-NEXT: %a = load {}, ptr %p, align 1
  ; CHECK-NEXT: store {} zeroinitializer, ptr @__dfsan_retval_tls, align [[ALIGN:2]]
  ; CHECK-NEXT: store i32 0, ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT: ret {} %a

  %a = load {}, ptr %p
  ret {} %a
}

define i16 @load_non_escaped_alloca() {
  ; CHECK-LABEL: @load_non_escaped_alloca.dfsan
  ; CHECK-NEXT: %[[#S_ALLOCA:]] = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK-NEXT: %_dfsa = alloca i32, align 4
  ; CHECK:      %[[#SHADOW:]] = load i[[#SBITS]], ptr %[[#S_ALLOCA]], align [[#SBYTES]]
  ; CHECK-NEXT: %[[#ORIGIN:]] = load i32, ptr %_dfsa, align 4
  ; CHECK-NEXT: %a = load i16, ptr %p, align 2
  ; CHECK-NEXT: store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT: store i32 %[[#ORIGIN]], ptr @__dfsan_retval_origin_tls, align 4

  %p = alloca i16
  %a = load i16, ptr %p
  ret i16 %a
}

define ptr @load_escaped_alloca() {
  ; CHECK-LABEL:  @load_escaped_alloca.dfsan
  ; CHECK:        %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:   %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:   %[[#SHADOW_PTR0:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:   %[[#ORIGIN_OFFSET:]] = add i64 %[[#SHADOW_OFFSET]], [[#%.10d,ORIGIN_BASE:]]
  ; CHECK-NEXT:   %[[#ORIGIN_ADDR:]] = and i64 %[[#ORIGIN_OFFSET]], -4
  ; CHECK-NEXT:   %[[#ORIGIN_PTR:]] = inttoptr i64 %[[#ORIGIN_ADDR]] to ptr
  ; CHECK-NEXT:   {{%.*}} = load i32, ptr %[[#ORIGIN_PTR]], align 4
  ; CHECK-NEXT:   %[[#SHADOW_PTR1:]] = getelementptr i[[#SBITS]], ptr %[[#SHADOW_PTR0]], i64 1
  ; CHECK-NEXT:   %[[#SHADOW:]]  = load i[[#SBITS]], ptr %[[#SHADOW_PTR0]], align [[#SBYTES]]
  ; CHECK-NEXT:   %[[#SHADOW+1]] = load i[[#SBITS]], ptr %[[#SHADOW_PTR1]], align [[#SBYTES]]
  ; CHECK-NEXT:   {{%.*}} = or i[[#SBITS]] %[[#SHADOW]], %[[#SHADOW+1]]
  ; CHECK-NEXT:   %a = load i16, ptr %p, align 2
  ; CHECK-NEXT:   store i[[#SBITS]] 0, ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:   store i32 0, ptr @__dfsan_retval_origin_tls, align 4

  %p = alloca i16
  %a = load i16, ptr %p
  ret ptr %p
}

@X = constant i1 1
define i1 @load_global() {
  ; CHECK-LABEL: @load_global.dfsan
  ; CHECK: %a = load i1, ptr @X, align 1
  ; CHECK-NEXT: store i[[#SBITS]] 0, ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT: store i32 0, ptr @__dfsan_retval_origin_tls, align 4

  %a = load i1, ptr @X
  ret i1 %a
}

define i1 @load1(ptr %p) {
  ; CHECK-LABEL:             @load1.dfsan

  ; COMBINE_LOAD_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#ORIGIN_OFFSET:]] = add i64 %[[#SHADOW_OFFSET]], [[#ORIGIN_BASE]]
  ; CHECK-NEXT:            %[[#ORIGIN_ADDR:]] = and i64 %[[#ORIGIN_OFFSET]], -4
  ; CHECK-NEXT:            %[[#ORIGIN_PTR:]] = inttoptr i64 %[[#ORIGIN_ADDR]] to ptr
  ; CHECK-NEXT:            %[[#AO:]] = load i32, ptr %[[#ORIGIN_PTR]], align 4
  ; CHECK-NEXT:            %[[#AS:]] = load i[[#SBITS]], ptr %[[#SHADOW_PTR]], align [[#SBYTES]]

  ; COMBINE_LOAD_PTR-NEXT: %[[#AS:]] = or i[[#SBITS]] %[[#AS]], %[[#PS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_LOAD_PTR-NEXT: %[[#AO:]] = select i1 %[[#NZ]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK-NEXT:            %a = load i1, ptr %p, align 1
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#AS]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            store i32 %[[#AO]], ptr @__dfsan_retval_origin_tls, align 4

  %a = load i1, ptr %p
  ret i1 %a
}

define i16 @load16(i1 %i, ptr %p) {
  ; CHECK-LABEL: @load16.dfsan

  ; COMBINE_LOAD_PTR-NEXT: %[[#PO:]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]

  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR0:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#ORIGIN_OFFSET:]] = add i64 %[[#SHADOW_OFFSET]], [[#ORIGIN_BASE]]
  ; CHECK-NEXT:            %[[#ORIGIN_ADDR:]] = and i64 %[[#ORIGIN_OFFSET]], -4
  ; CHECK-NEXT:            %[[#ORIGIN_PTR:]] = inttoptr i64 %[[#ORIGIN_ADDR]] to ptr
  ; CHECK-NEXT:            %[[#AO:]] = load i32, ptr %[[#ORIGIN_PTR]], align 4
  ; CHECK-NEXT:            %[[#SHADOW_PTR1:]] = getelementptr i[[#SBITS]], ptr %[[#SHADOW_PTR0]], i64 1
  ; CHECK-NEXT:            %[[#SHADOW:]]  = load i[[#SBITS]], ptr %[[#SHADOW_PTR0]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#SHADOW+1]] = load i[[#SBITS]], ptr %[[#SHADOW_PTR1]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#AS:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#SHADOW+1]]

  ; COMBINE_LOAD_PTR-NEXT: %[[#AS:]] = or i[[#SBITS]] %[[#AS]], %[[#PS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_LOAD_PTR-NEXT: %[[#AO:]] = select i1 %[[#NZ]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK-NEXT:            %a = load i16, ptr %p, align 2
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#AS]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            store i32 %[[#AO]], ptr @__dfsan_retval_origin_tls, align 4

  %a = load i16, ptr %p
  ret i16 %a
}

define i32 @load32(ptr %p) {
  ; CHECK-LABEL: @load32.dfsan

  ; COMBINE_LOAD_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#ORIGIN_ADDR:]] = add i64 %[[#SHADOW_OFFSET]], [[#ORIGIN_BASE]]
  ; CHECK-NEXT:            %[[#ORIGIN_PTR:]] = inttoptr i64 %[[#ORIGIN_ADDR]] to ptr
  ; CHECK-NEXT:            %[[#AO:]] = load i32, ptr %[[#ORIGIN_PTR]], align 4
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = load i[[#WSBITS:mul(SBITS,4)]], ptr %[[#SHADOW_PTR]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW+1]] = lshr i[[#WSBITS]] %[[#WIDE_SHADOW]], [[#mul(SBITS,2)]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW+2]] = or i[[#WSBITS]] %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW+1]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW+3]] = lshr i[[#WSBITS]] %[[#WIDE_SHADOW+2]], [[#SBITS]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW+4]] = or i[[#WSBITS]] %[[#WIDE_SHADOW+2]], %[[#WIDE_SHADOW+3]]
  ; CHECK-NEXT:            %[[#SHADOW:]] = trunc i[[#WSBITS]] %[[#WIDE_SHADOW+4]] to i[[#SBITS]]

  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_LOAD_PTR-NEXT: %[[#AO:]] = select i1 %[[#NZ]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK-NEXT:            %a = load i32, ptr %p, align 4
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            store i32 %[[#AO]], ptr @__dfsan_retval_origin_tls, align 4

  %a = load i32, ptr %p
  ret i32 %a
}

define i64 @load64(ptr %p) {
  ; CHECK-LABEL: @load64.dfsan

  ; COMBINE_LOAD_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#ORIGIN_ADDR:]] = add i64 %[[#SHADOW_OFFSET]], [[#ORIGIN_BASE]]
  ; CHECK-NEXT:            %[[#ORIGIN_PTR:]] = inttoptr i64 %[[#ORIGIN_ADDR]] to ptr
  ; CHECK-NEXT:            %[[#ORIGIN:]] = load i32, ptr %[[#ORIGIN_PTR]], align 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = load i64, ptr %[[#SHADOW_PTR]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_LO:]] = shl i64 %[[#WIDE_SHADOW]], 32
  ; CHECK-NEXT:            %[[#ORIGIN2_PTR:]] = getelementptr i32, ptr %[[#ORIGIN_PTR]], i64 1
  ; CHECK-NEXT:            %[[#ORIGIN2:]] = load i32, ptr %[[#ORIGIN2_PTR]], align 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 32
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 16
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#SHADOW:]] = trunc i64 %[[#WIDE_SHADOW]] to i[[#SBITS]]
  ; CHECK-NEXT:            %[[#SHADOW_NZ:]] = icmp ne i64 %[[#WIDE_SHADOW_LO]], 0
  ; CHECK-NEXT:            %[[#ORIGIN:]] = select i1 %[[#SHADOW_NZ]], i32 %[[#ORIGIN]], i32 %[[#ORIGIN2]]
  ; CHECK8-NEXT:           %[[#SHADOW_NZ:]] = icmp ne i64 %[[#WIDE_SHADOW_LO]], 0
  ; CHECK8-NEXT:           %[[#ORIGIN:]] = select i1 %[[#SHADOW_NZ]], i32 %[[#ORIGIN]], i32 %[[#ORIGIN2]]

  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_LOAD_PTR-NEXT: %[[#ORIGIN:]] = select i1 %[[#NZ]], i32 %[[#PO]], i32 %[[#ORIGIN]]

  ; CHECK-NEXT:            %a = load i64, ptr %p, align 8
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            store i32 %[[#ORIGIN]], ptr @__dfsan_retval_origin_tls, align 4

  %a = load i64, ptr %p
  ret i64 %a
}

define i64 @load64_align2(ptr %p) {
  ; CHECK-LABEL: @load64_align2.dfsan

  ; COMBINE_LOAD_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT:            %[[#LABEL_ORIGIN:]] = call zeroext i64 @__dfsan_load_label_and_origin(ptr %p, i64 8)
  ; CHECK-NEXT:            %[[#LABEL_ORIGIN+1]] = lshr i64 %[[#LABEL_ORIGIN]], 32
  ; CHECK-NEXT:            %[[#LABEL:]] = trunc i64 %[[#LABEL_ORIGIN+1]] to i[[#SBITS]]
  ; CHECK-NEXT:            %[[#ORIGIN:]] = trunc i64 %[[#LABEL_ORIGIN]] to i32

  ; COMBINE_LOAD_PTR-NEXT: %[[#LABEL:]] = or i[[#SBITS]] %[[#LABEL]], %[[#PS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_LOAD_PTR-NEXT: %[[#ORIGIN:]] = select i1 %[[#NZ]], i32 %[[#PO]], i32 %[[#ORIGIN]]

  ; CHECK-NEXT:            %a = load i64, ptr %p, align 2
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#LABEL]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            store i32 %[[#ORIGIN]], ptr @__dfsan_retval_origin_tls, align 4

  %a = load i64, ptr %p, align 2
  ret i64 %a
}

define i128 @load128(ptr %p) {
  ; CHECK-LABEL: @load128.dfsan

  ; COMBINE_LOAD_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT:            %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:            %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#MASK]]
  ; CHECK-NEXT:            %[[#SHADOW_PTR:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:            %[[#ORIGIN_ADDR:]] = add i64 %[[#SHADOW_OFFSET]], [[#ORIGIN_BASE]]
  ; CHECK-NEXT:            %[[#ORIGIN1_PTR:]] = inttoptr i64 %[[#ORIGIN_ADDR]] to ptr
  ; CHECK-NEXT:            %[[#ORIGIN1:]] = load i32, ptr %[[#ORIGIN1_PTR]], align 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW1:]] = load i64, ptr %[[#SHADOW_PTR]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW1_LO:]] = shl i64 %[[#WIDE_SHADOW1]], 32
  ; CHECK-NEXT:            %[[#ORIGIN2_PTR:]] = getelementptr i32, ptr %[[#ORIGIN1_PTR]], i64 1
  ; CHECK-NEXT:            %[[#ORIGIN2:]] = load i32, ptr %[[#ORIGIN2_PTR]], align 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW2_PTR:]] = getelementptr i64, ptr %[[#SHADOW_PTR]], i64 1
  ; CHECK-NEXT:            %[[#WIDE_SHADOW2:]] = load i64, ptr %[[#WIDE_SHADOW2_PTR]], align [[#SBYTES]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW1]], %[[#WIDE_SHADOW2]]
  ; CHECK-NEXT:            %[[#ORIGIN3_PTR:]] = getelementptr i32, ptr %[[#ORIGIN2_PTR]], i64 1
  ; CHECK-NEXT:            %[[#ORIGIN3:]] = load i32, ptr %[[#ORIGIN3_PTR]], align 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW2_LO:]] = shl i64 %[[#WIDE_SHADOW2]], 32
  ; CHECK-NEXT:            %[[#ORIGIN4_PTR:]] = getelementptr i32, ptr %[[#ORIGIN3_PTR]], i64 1
  ; CHECK-NEXT:            %[[#ORIGIN4:]] = load i32, ptr %[[#ORIGIN4_PTR]], align 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 32
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 16
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#WIDE_SHADOW_SHIFTED:]] = lshr i64 %[[#WIDE_SHADOW]], 8
  ; CHECK-NEXT:            %[[#WIDE_SHADOW:]] = or i64 %[[#WIDE_SHADOW]], %[[#WIDE_SHADOW_SHIFTED]]
  ; CHECK-NEXT:            %[[#SHADOW:]] = trunc i64 %[[#WIDE_SHADOW]] to i[[#SBITS]]
  ; CHECK-NEXT:            %[[#SHADOW1_LO_NZ:]] = icmp ne i64 %[[#WIDE_SHADOW1_LO]], 0
  ; CHECK-NEXT:            %[[#ORIGIN12:]] = select i1 %[[#SHADOW1_LO_NZ]], i32 %[[#ORIGIN1]], i32 %[[#ORIGIN2]]
  ; CHECK-NEXT:            %[[#SHADOW2_NZ:]] = icmp ne i64 %[[#WIDE_SHADOW2]], 0
  ; CHECK-NEXT:            %[[#ORIGIN124:]] = select i1 %[[#SHADOW2_NZ]], i32 %[[#ORIGIN4]], i32 %[[#ORIGIN12]]
  ; CHECK-NEXT:            %[[#SHADOW2_LO_NZ:]] = icmp ne i64 %[[#WIDE_SHADOW2_LO]], 0
  ; CHECK-NEXT:            %[[#ORIGIN:]] = select i1 %[[#SHADOW2_LO_NZ]], i32 %[[#ORIGIN3]], i32 %[[#ORIGIN124]]

  ; COMBINE_LOAD_PTR-NEXT: %[[#SHADOW:]] = or i[[#SBITS]] %[[#SHADOW]], %[[#PS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_LOAD_PTR-NEXT: %[[#ORIGIN:]] = select i1 %[[#NZ]], i32 %[[#PO]], i32 %[[#ORIGIN]]

  ; CHECK-NEXT:            %a = load i128, ptr %p, align 8
  ; CHECK-NEXT:            store i[[#SBITS]] %[[#SHADOW]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT:            store i32 %[[#ORIGIN]], ptr @__dfsan_retval_origin_tls, align 4

  %a = load i128, ptr %p
  ret i128 %a
}

define i17 @load17(ptr %p) {
  ; CHECK-LABEL: @load17.dfsan

  ; COMBINE_LOAD_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_LOAD_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT: %[[#LABEL_ORIGIN:]] = call zeroext i64 @__dfsan_load_label_and_origin(ptr %p, i64 3)
  ; CHECK-NEXT: %[[#LABEL_ORIGIN_H32:]] = lshr i64 %[[#LABEL_ORIGIN]], 32
  ; CHECK-NEXT: %[[#LABEL:]] = trunc i64 %[[#LABEL_ORIGIN_H32]] to i[[#SBITS]]
  ; CHECK-NEXT: %[[#ORIGIN:]] = trunc i64 %[[#LABEL_ORIGIN]] to i32

  ; COMBINE_LOAD_PTR-NEXT: %[[#LABEL:]] = or i[[#SBITS]] %[[#LABEL]], %[[#PS]]
  ; COMBINE_LOAD_PTR-NEXT: %[[#NZ:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_LOAD_PTR-NEXT: %[[#ORIGIN:]] = select i1 %[[#NZ]], i32 %[[#PO]], i32 %[[#ORIGIN]]

  ; CHECK-NEXT: %a = load i17, ptr %p, align 4
  ; CHECK-NEXT: store i[[#SBITS]] %[[#LABEL]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK-NEXT: store i32 %[[#ORIGIN]], ptr @__dfsan_retval_origin_tls, align 4

  %a = load i17, ptr %p, align 4
  ret i17 %a
}
