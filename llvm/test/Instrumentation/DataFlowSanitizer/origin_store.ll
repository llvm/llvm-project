; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-8-labels -S | FileCheck %s
; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-8-labels -dfsan-combine-pointer-labels-on-store -S | FileCheck %s --check-prefixes=CHECK,COMBINE_STORE_PTR
; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels -S | FileCheck %s --check-prefixes=CHECK,CHECK16
; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels -dfsan-combine-pointer-labels-on-store -S | FileCheck %s --check-prefixes=CHECK,CHECK16,COMBINE_STORE_PTR
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define void @store_zero_to_non_escaped_alloca() {
  ; CHECK-LABEL: @"dfs$store_zero_to_non_escaped_alloca"
  ; CHECK-NEXT: [[A:%.*]] = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK-NEXT: %_dfsa = alloca i32, align 4
  ; CHECK-NEXT: %p = alloca i16, align 2
  ; CHECK-NEXT: store i[[#SBITS]] 0, i[[#SBITS]]* [[A]], align [[#SBYTES]]
  ; CHECK-NEXT: store i16 1, i16* %p, align 2
  ; CHECK-NEXT: ret void
  
  %p = alloca i16
  store i16 1, i16* %p
  ret void
}

define void @store_nonzero_to_non_escaped_alloca(i16 %a) {
  ; CHECK-LABEL: @"dfs$store_nonzero_to_non_escaped_alloca"
  ; CHECK: %[[#AO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: %_dfsa = alloca i32, align 4
  ; CHECK: store i32 %[[#AO]], i32* %_dfsa, align 4
  
  %p = alloca i16
  store i16 %a, i16* %p
  ret void
}

declare void @foo(i16* %p)

define void @store_zero_to_escaped_alloca() {
  ; CHECK-LABEL: @"dfs$store_zero_to_escaped_alloca"
  ; CHECK:       %[[#SA:]] = bitcast i[[#SBITS]]* {{.*}} to i[[#NUM_BITS:mul(SBITS,2)]]*
  ; CHECK-NEXT:  store i[[#NUM_BITS]] 0, i[[#NUM_BITS]]* %[[#SA]], align [[#SBYTES]]
  ; CHECK-NEXT:  store i16 1, i16* %p, align 2
  ; CHECK-NEXT:  store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; CHECK-NEXT:  call void @"dfs$foo"(i16* %p)

  %p = alloca i16
  store i16 1, i16* %p
  call void @foo(i16* %p)
  ret void
}

define void @store_nonzero_to_escaped_alloca(i16 %a) {
  ; CHECK-LABEL:  @"dfs$store_nonzero_to_escaped_alloca"
  ; CHECK-NEXT:   %[[#AO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK-NEXT:   %[[#AS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; CHECK:        %[[#INTP:]] = ptrtoint i16* %p to i64
  ; CHECK-NEXT:   %[[#SHADOW_ADDR:]] = and i64 %[[#INTP]], [[#%.10d,MASK:]]
  ; CHECK16-NEXT: %[[#SHADOW_ADDR:]] = mul i64 %[[#SHADOW_ADDR]], 2
  ; CHECK-NEXT:   %[[#SHADOW_PTR0:]] = inttoptr i64 %[[#SHADOW_ADDR]] to i[[#SBITS]]*
  ; CHECK-NEXT:   %[[#ORIGIN_OFFSET:]] = add i64 %[[#INTP+1]], [[#%.10d,ORIGIN_MASK:]]
  ; CHECK-NEXT:   %[[#ORIGIN_ADDR:]] = and i64 %[[#ORIGIN_OFFSET]], -4
  ; CHECK-NEXT:   %[[#ORIGIN_PTR:]] = inttoptr i64 %[[#ORIGIN_ADDR]] to i32*
  ; CHECK:        %_dfscmp = icmp ne i[[#SBITS]] %[[#AS]], 0
  ; CHECK-NEXT:   br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK:       [[L1]]:
  ; CHECK-NEXT:   %[[#NO:]] = call zeroext i32 @__dfsan_chain_origin(i32 zeroext %[[#AO]])
  ; CHECK-NEXT:   store i32 %[[#NO]], i32* %[[#ORIGIN_PTR]], align 4
  ; CHECK-NEXT:   br label %[[L2]]
  ; CHECK:       [[L2]]:
  ; CHECK-NEXT:    store i16 %a, i16* %p, align 2
  
  %p = alloca i16
  store i16 %a, i16* %p
  call void @foo(i16* %p)
  ret void
}

define void @store64_align8(i64* %p, i64 %a) {
  ; CHECK-LABEL: @"dfs$store64_align8"

  ; COMBINE_STORE_PTR-NEXT: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; COMBINE_STORE_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]

  ; CHECK-NEXT:  %[[#AO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT:  %[[#AS:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]

  ; COMBINE_STORE_PTR-NEXT: %[[#AS:]] = or i[[#SBITS]] %[[#AS]], %[[#PS]]
  ; COMBINE_STORE_PTR-NEXT: %[[#NE:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_STORE_PTR-NEXT: %[[#AO:]] = select i1 %[[#NE]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK:       %_dfscmp = icmp ne i[[#SBITS]] %[[#AS]], 0
  ; CHECK-NEXT:  br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK:      [[L1]]:
  ; CHECK-NEXT:  %[[#NO:]] = call zeroext i32 @__dfsan_chain_origin(i32 zeroext %[[#AO]])
  ; CHECK-NEXT:  %[[#NO_ZEXT:]] = zext i32 %[[#NO]] to i64
  ; CHECK-NEXT:  %[[#NO_SHL:]] = shl i64 %[[#NO_ZEXT]], 32
  ; CHECK-NEXT:  %[[#NO2:]] = or i64 %[[#NO_ZEXT]], %[[#NO_SHL]]
  ; CHECK-NEXT:  %[[#O_PTR:]] = bitcast i32* {{.*}} to i64*
  ; CHECK-NEXT:  store i64 %[[#NO2]], i64* %[[#O_PTR]], align 8
  ; CHECK-NEXT:  br label %[[L2]]
  ; CHECK:      [[L2]]:
  ; CHECK-NEXT:  store i64 %a, i64* %p, align 8
  
  store i64 %a, i64* %p
  ret void
}

define void @store64_align2(i64* %p, i64 %a) {
  ; CHECK-LABEL: @"dfs$store64_align2"

  ; COMBINE_STORE_PTR-NEXT: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; COMBINE_STORE_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]

  ; CHECK-NEXT: %[[#AO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT: %[[#AS:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]

  ; COMBINE_STORE_PTR-NEXT: %[[#AS:]] = or i[[#SBITS]] %[[#AS]], %[[#PS]]
  ; COMBINE_STORE_PTR-NEXT: %[[#NE:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_STORE_PTR-NEXT: %[[#AO:]] = select i1 %[[#NE]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK:      %_dfscmp = icmp ne i[[#SBITS]] %[[#AS]], 0
  ; CHECK-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK:     [[L1]]:
  ; CHECK-NEXT: %[[#NO:]] = call zeroext i32 @__dfsan_chain_origin(i32 zeroext %[[#AO]])
  ; CHECK-NEXT: store i32 %[[#NO]], i32* %[[#O_PTR0:]], align 4
  ; CHECK-NEXT: %[[#O_PTR1:]] = getelementptr i32, i32* %[[#O_PTR0]], i32 1
  ; CHECK-NEXT: store i32 %[[#NO]], i32* %[[#O_PTR1]], align 4
  ; CHECK:     [[L2]]:
  ; CHECK-NEXT: store i64 %a, i64* %p, align 2
  
  store i64 %a, i64* %p, align 2
  ret void
}

define void @store96_align8(i96* %p, i96 %a) {
  ; CHECK-LABEL: @"dfs$store96_align8"

  ; COMBINE_STORE_PTR-NEXT: %[[#PO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; COMBINE_STORE_PTR-NEXT: %[[#PS:]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]

  ; CHECK-NEXT: %[[#AO:]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT: %[[#AS:]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN]]

  ; COMBINE_STORE_PTR-NEXT: %[[#AS:]] = or i[[#SBITS]] %[[#AS]], %[[#PS]]
  ; COMBINE_STORE_PTR-NEXT: %[[#NE:]] = icmp ne i[[#SBITS]] %[[#PS]], 0
  ; COMBINE_STORE_PTR-NEXT: %[[#AO:]] = select i1 %[[#NE]], i32 %[[#PO]], i32 %[[#AO]]

  ; CHECK:      %_dfscmp = icmp ne i[[#SBITS]] %[[#AS]], 0
  ; CHECK-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK:     [[L1]]:
  ; CHECK-NEXT: %[[#NO:]] = call zeroext i32 @__dfsan_chain_origin(i32 zeroext %[[#AO]])
  ; CHECK-NEXT: %[[#NO_ZEXT:]] = zext i32 %[[#NO]] to i64
  ; CHECK-NEXT: %[[#NO_SHL:]] = shl i64 %[[#NO_ZEXT]], 32
  ; CHECK-NEXT: %[[#NO2:]] = or i64 %[[#NO_ZEXT]], %[[#NO_SHL]]
  ; CHECK-NEXT: %[[#O_PTR64:]] = bitcast i32* %[[#O_PTR0:]] to i64*
  ; CHECK-NEXT: store i64 %[[#NO2]], i64* %[[#O_PTR64]], align 8
  ; CHECK-NEXT: %[[#O_PTR1:]] = getelementptr i32, i32* %[[#O_PTR0]], i32 2
  ; CHECK-NEXT: store i32 %[[#NO]], i32* %[[#O_PTR1]], align 8
  ; CHECK:     [[L2]]:
  ; CHECK-NEXT: store i96 %a, i96* %p, align 8
  
  store i96 %a, i96* %p, align 8
  ret void
}
