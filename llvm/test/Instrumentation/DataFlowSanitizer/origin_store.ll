; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1 -S | FileCheck %s
; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1 -dfsan-combine-pointer-labels-on-store -S | FileCheck %s --check-prefixes=CHECK,COMBINE_STORE_PTR
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define void @store_zero_to_non_escaped_alloca() {
  ; CHECK-LABEL: @store_zero_to_non_escaped_alloca.dfsan
  ; CHECK-NEXT: [[A:%.*]] = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK-NEXT: %_dfsa = alloca i32, align 4
  ; CHECK-NEXT: %p = alloca i16, align 2
  ; CHECK-NEXT: store i[[#SBITS]] 0, ptr [[A]], align [[#SBYTES]]
  ; CHECK-NEXT: store i16 1, ptr %p, align 2
  ; CHECK-NEXT: ret void

  %p = alloca i16
  store i16 1, ptr %p
  ret void
}

define void @store_nonzero_to_non_escaped_alloca(i16 %a) {
  ; CHECK-LABEL: @store_nonzero_to_non_escaped_alloca.dfsan
  ; CHECK: %[[#AO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: %_dfsa = alloca i32, align 4
  ; CHECK: store i32 %[[#AO]], ptr %_dfsa, align 4

  %p = alloca i16
  store i16 %a, ptr %p
  ret void
}

declare void @foo(ptr %p)

define void @store_zero_to_escaped_alloca() {
  ; CHECK-LABEL: @store_zero_to_escaped_alloca.dfsan
  ; CHECK:  store i[[#NUM_BITS:mul(SBITS,2)]] 0, ptr {{.*}}, align [[#SBYTES]]
  ; CHECK-NEXT:  store i16 1, ptr %p, align 2
  ; CHECK-NEXT:  store i[[#SBITS]] 0, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; CHECK-NEXT:  call void @foo.dfsan(ptr %p)

  %p = alloca i16
  store i16 1, ptr %p
  call void @foo(ptr %p)
  ret void
}

define void @store_nonzero_to_escaped_alloca(i16 %a) {
  ; CHECK-LABEL:  @store_nonzero_to_escaped_alloca.dfsan
  ; CHECK-NEXT:   %[[#AO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK-NEXT:   %[[#AS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK:        %[[#INTP:]] = ptrtoint ptr %p to i64
  ; CHECK-NEXT:   %[[#SHADOW_OFFSET:]] = xor i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK-NEXT:   %[[#SHADOW_PTR0:]] = inttoptr i64 %[[#SHADOW_OFFSET]] to ptr
  ; CHECK-NEXT:   %[[#ORIGIN_OFFSET:]] = add i64 %[[#SHADOW_OFFSET]], [[#%.10d,ORIGIN_BASE:]]
  ; CHECK-NEXT:   %[[#ORIGIN_ADDR:]] = and i64 %[[#ORIGIN_OFFSET]], -4
  ; CHECK-NEXT:   %[[#ORIGIN_PTR:]] = inttoptr i64 %[[#ORIGIN_ADDR]] to ptr
  ; CHECK:        %_dfscmp = icmp ne i[[#SBITS]] %[[#AS]], 0
  ; CHECK-NEXT:   br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK:       [[L1]]:
  ; CHECK-NEXT:   %[[#NO:]] = call i32 @__dfsan_chain_origin(i32 %[[#AO]])
  ; CHECK-NEXT:   store i32 %[[#NO]], ptr %[[#ORIGIN_PTR]], align 4
  ; CHECK-NEXT:   br label %[[L2]]
  ; CHECK:       [[L2]]:
  ; CHECK-NEXT:    store i16 %a, ptr %p, align 2

  %p = alloca i16
  store i16 %a, ptr %p
  call void @foo(ptr %p)
  ret void
}

define void @store64_align8(ptr %p, i64 %a) {
  ; CHECK-LABEL: @store64_align8.dfsan

  ; COMBINE_STORE_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_STORE_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT:  %[[#AO:]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT:  %[[#AS:]] = load i[[#SBITS]], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]

  ; COMBINE_STORE_PTR-NEXT: %[[#AS:]] = or i[[#SBITS]] %[[#AS]], %[[#PS]]
  ; COMBINE_STORE_PTR-NEXT: %[[#NE:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_STORE_PTR-NEXT: %[[#AO:]] = select i1 %[[#NE]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK:       %_dfscmp = icmp ne i[[#SBITS]] %[[#AS]], 0
  ; CHECK-NEXT:  br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK:      [[L1]]:
  ; CHECK-NEXT:  %[[#NO:]] = call i32 @__dfsan_chain_origin(i32 %[[#AO]])
  ; CHECK-NEXT:  %[[#NO_ZEXT:]] = zext i32 %[[#NO]] to i64
  ; CHECK-NEXT:  %[[#NO_SHL:]] = shl i64 %[[#NO_ZEXT]], 32
  ; CHECK-NEXT:  %[[#NO2:]] = or i64 %[[#NO_ZEXT]], %[[#NO_SHL]]
  ; CHECK-NEXT:  store i64 %[[#NO2]], ptr {{.*}}, align 8
  ; CHECK-NEXT:  br label %[[L2]]
  ; CHECK:      [[L2]]:
  ; CHECK-NEXT:  store i64 %a, ptr %p, align 8

  store i64 %a, ptr %p
  ret void
}

define void @store64_align2(ptr %p, i64 %a) {
  ; CHECK-LABEL: @store64_align2.dfsan

  ; COMBINE_STORE_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_STORE_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT: %[[#AO:]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT: %[[#AS:]] = load i[[#SBITS]], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]

  ; COMBINE_STORE_PTR-NEXT: %[[#AS:]] = or i[[#SBITS]] %[[#AS]], %[[#PS]]
  ; COMBINE_STORE_PTR-NEXT: %[[#NE:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_STORE_PTR-NEXT: %[[#AO:]] = select i1 %[[#NE]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK:      %_dfscmp = icmp ne i[[#SBITS]] %[[#AS]], 0
  ; CHECK-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK:     [[L1]]:
  ; CHECK-NEXT: %[[#NO:]] = call i32 @__dfsan_chain_origin(i32 %[[#AO]])
  ; CHECK-NEXT: store i32 %[[#NO]], ptr %[[#O_PTR0:]], align 4
  ; CHECK-NEXT: %[[#O_PTR1:]] = getelementptr i32, ptr %[[#O_PTR0]], i32 1
  ; CHECK-NEXT: store i32 %[[#NO]], ptr %[[#O_PTR1]], align 4
  ; CHECK:     [[L2]]:
  ; CHECK-NEXT: store i64 %a, ptr %p, align 2

  store i64 %a, ptr %p, align 2
  ret void
}

define void @store96_align8(ptr %p, i96 %a) {
  ; CHECK-LABEL: @store96_align8.dfsan

  ; COMBINE_STORE_PTR-NEXT: %[[#PO:]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMBINE_STORE_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], ptr @__dfsan_arg_tls, align [[ALIGN]]

  ; CHECK-NEXT: %[[#AO:]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT: %[[#AS:]] = load i[[#SBITS]], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]

  ; COMBINE_STORE_PTR-NEXT: %[[#AS:]] = or i[[#SBITS]] %[[#AS]], %[[#PS]]
  ; COMBINE_STORE_PTR-NEXT: %[[#NE:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_STORE_PTR-NEXT: %[[#AO:]] = select i1 %[[#NE]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK:      %_dfscmp = icmp ne i[[#SBITS]] %[[#AS]], 0
  ; CHECK-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK:     [[L1]]:
  ; CHECK-NEXT: %[[#NO:]] = call i32 @__dfsan_chain_origin(i32 %[[#AO]])
  ; CHECK-NEXT: %[[#NO_ZEXT:]] = zext i32 %[[#NO]] to i64
  ; CHECK-NEXT: %[[#NO_SHL:]] = shl i64 %[[#NO_ZEXT]], 32
  ; CHECK-NEXT: %[[#NO2:]] = or i64 %[[#NO_ZEXT]], %[[#NO_SHL]]
  ; CHECK-NEXT: store i64 %[[#NO2]], ptr %[[#O_PTR0:]], align 8
  ; CHECK-NEXT: %[[#O_PTR1:]] = getelementptr i32, ptr %[[#O_PTR0]], i32 2
  ; CHECK-NEXT: store i32 %[[#NO]], ptr %[[#O_PTR1]], align 8
  ; CHECK:     [[L2]]:
  ; CHECK-NEXT: store i96 %a, ptr %p, align 8

  store i96 %a, ptr %p, align 8
  ret void
}
