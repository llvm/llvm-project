; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefixes=CHECK_META,CHECK
; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s --check-prefixes=CHECK_META,NO_COMBINE_LOAD_PTR
; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-store=true -S | FileCheck %s --check-prefixes=CHECK_META,COMBINE_STORE_PTR
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK_META: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK_META: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define {} @load0({}* %p) {
  ; CHECK-LABEL: @"dfs$load0"
  ; CHECK-NEXT: %a = load {}, {}* %p, align 1
  ; CHECK-NEXT: store {} zeroinitializer, {}* bitcast ([100 x i64]* @__dfsan_retval_tls to {}*), align [[#SBYTES]]
  ; CHECK-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT: ret {} %a

  %a = load {}, {}* %p
  ret {} %a
}

define i16 @load_non_escaped_alloca() {
  ; CHECK-LABEL: @"dfs$load_non_escaped_alloca"
  ; CHECK-NEXT: [[S_ALLOCA:%.*]] = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK-NEXT: [[O_ALLOCA:%.*]] = alloca i32, align 4
  ; CHECK: [[SHADOW:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[S_ALLOCA]], align [[#SBYTES]]
  ; CHECK-NEXT: [[ORIGIN:%.*]] = load i32, i32* [[O_ALLOCA]], align 4
  ; CHECK-NEXT: %a = load i16, i16* %p, align 2
  ; CHECK-NEXT: store i[[#SBITS]] [[SHADOW]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK-NEXT: store i32 [[ORIGIN]], i32* @__dfsan_retval_origin_tls, align 4
  
  %p = alloca i16
  %a = load i16, i16* %p
  ret i16 %a
}

define i16* @load_escaped_alloca() {
  ; CHECK-LABEL: @"dfs$load_escaped_alloca"
  ; CHECK: [[INTP:%.*]] = ptrtoint i[[#SBITS]]* %p to i64
  ; CHECK-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]]
  ; CHECK-NEXT: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; CHECK-NEXT: [[SHADOW_PTR0:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; CHECK-NEXT: [[ORIGIN_OFFSET:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; CHECK-NEXT: [[ORIGIN_ADDR:%.*]] = and i64 [[ORIGIN_OFFSET]], -4
  ; CHECK-NEXT: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; CHECK-NEXT: {{%.*}} = load i32, i32* [[ORIGIN_PTR]], align 4
  ; CHECK-NEXT: [[SHADOW_PTR1:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR0]], i64 1
  ; CHECK-NEXT: [[SHADOW0:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR0]], align [[#SBYTES]]
  ; CHECK-NEXT: [[SHADOW1:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR1]], align [[#SBYTES]]
  ; CHECK-NEXT: {{%.*}} = or i[[#SBITS]] [[SHADOW0]], [[SHADOW1]]
  ; CHECK-NEXT: %a = load i16, i16* %p, align 2
  ; CHECK-NEXT: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  
  %p = alloca i16
  %a = load i16, i16* %p
  ret i16* %p
}

@X = constant i1 1
define i1 @load_global() {
  ; CHECK-LABEL: @"dfs$load_global"
  ; CHECK: %a = load i1, i1* @X, align 1
  ; CHECK-NEXT: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4

  %a = load i1, i1* @X
  ret i1 %a
}

define i1 @load1(i1* %p) {
  ; CHECK-LABEL: @"dfs$load1"
  ; CHECK-NEXT: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK-NEXT: [[PS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK-NEXT: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
  ; CHECK-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]]
  ; CHECK-NEXT: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; CHECK-NEXT: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; CHECK-NEXT: [[ORIGIN_OFFSET:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; CHECK-NEXT: [[ORIGIN_ADDR:%.*]] = and i64 [[ORIGIN_OFFSET]], -4
  ; CHECK-NEXT: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; CHECK-NEXT: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 4
  ; CHECK-NEXT: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR]], align [[#SBYTES]]
  ; CHECK-NEXT: [[RS:%.*]] = or i[[#SBITS]] [[AS]], [[PS]]
  ; CHECK-NEXT: [[PS_NZ:%.*]] = icmp ne i[[#SBITS]] [[PS]], 0
  ; CHECK-NEXT: [[RO:%.*]] = select i1 [[PS_NZ]], i32 [[PO]], i32 [[AO]]
  ; CHECK-NEXT: %a = load i1, i1* %p, align 1
  ; CHECK-NEXT: store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK-NEXT: store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i1, i1* %p
  ret i1 %a
}

define i16 @load16(i1 %i, i16* %p) {
  ; CHECK-LABEL: @"dfs$load16"
  ; CHECK-NEXT: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT: [[PS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK-NEXT: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
  ; CHECK-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]]
  ; CHECK-NEXT: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; CHECK-NEXT: [[SHADOW_PTR0:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; CHECK-NEXT: [[ORIGIN_OFFSET:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; CHECK-NEXT: [[ORIGIN_ADDR:%.*]] = and i64 [[ORIGIN_OFFSET]], -4
  ; CHECK-NEXT: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; CHECK-NEXT: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 4
  ; CHECK-NEXT: [[SHADOW_PTR1:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR0]], i64 1
  ; CHECK-NEXT: [[SHADOW0:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR0]], align [[#SBYTES]]
  ; CHECK-NEXT: [[SHADOW1:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR1]], align [[#SBYTES]]
  ; CHECK-NEXT: [[AS:%.*]] = or i[[#SBITS]] [[SHADOW0]], [[SHADOW1]]
  ; CHECK-NEXT: [[RS:%.*]] = or i[[#SBITS]] [[AS]], [[PS]]
  ; CHECK-NEXT: [[PS_NZ:%.*]] = icmp ne i[[#SBITS]] [[PS]], 0
  ; CHECK-NEXT: [[RO:%.*]] = select i1 [[PS_NZ]], i32 [[PO]], i32 [[AO]]
  ; CHECK-NEXT: %a = load i16, i16* %p, align 2
  ; CHECK-NEXT: store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK-NEXT: store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i16, i16* %p
  ret i16 %a
}

define i32 @load32(i32* %p) {
  ; CHECK-LABEL: @"dfs$load32"

  ; NO_COMBINE_LOAD_PTR-LABEL: @"dfs$load32"
  ; NO_COMBINE_LOAD_PTR: [[INTP:%.*]] = ptrtoint i32* %p to i64
  ; NO_COMBINE_LOAD_PTR-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 4
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR64:%.*]] = bitcast i[[#SBITS]]* [[SHADOW_PTR]] to i64*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64:%.*]] = load i64, i64* [[SHADOW_PTR64]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR-NEXT: %a = load i32, i32* %p, align 4
  ; NO_COMBINE_LOAD_PTR-NEXT: store i[[#SBITS]] [[SHADOW]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: store i32 [[AO]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i32, i32* %p
  ret i32 %a
}

define i64 @load64(i64* %p) {
  ; CHECK-LABEL: @"dfs$load64"
  
  ; NO_COMBINE_LOAD_PTR-LABEL: @"dfs$load64"
  ; NO_COMBINE_LOAD_PTR: [[INTP:%.*]] = ptrtoint i64* %p to i64
  ; NO_COMBINE_LOAD_PTR-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_PTR_0:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_0:%.*]] = load i32, i32* [[ORIGIN_PTR_0]], align 8
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR_0:%.*]] = bitcast i[[#SBITS]]* [[SHADOW_PTR]] to i64*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_0:%.*]] = load i64, i64* [[SHADOW_PTR_0]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR_1:%.*]] = getelementptr i64, i64* [[SHADOW_PTR_0]], i64 1
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_1:%.*]] = load i64, i64* [[SHADOW_PTR_1]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64:%.*]] = or i64 [[SHADOW_0]], [[SHADOW_1]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_PTR_1:%.*]] = getelementptr i32, i32* [[ORIGIN_PTR_0]], i64 1
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_1:%.*]] = load i32, i32* [[ORIGIN_PTR_1]], align 8
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_1_NZ:%.*]] = icmp ne i64 [[SHADOW_1]], 0
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN:%.*]] = select i1 [[SHADOW_1_NZ]], i32 [[ORIGIN_1]], i32 [[ORIGIN_0]]
  ; NO_COMBINE_LOAD_PTR-NEXT: %a = load i64, i64* %p, align 8
  ; NO_COMBINE_LOAD_PTR-NEXT: store i[[#SBITS]] [[SHADOW]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: store i32 [[ORIGIN]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i64, i64* %p
  ret i64 %a
}

define i64 @load64_align2(i64* %p) {
  ; CHECK-LABEL: @"dfs$load64_align2"

  ; NO_COMBINE_LOAD_PTR-LABEL: @"dfs$load64_align2"
  ; NO_COMBINE_LOAD_PTR: [[INTP:%.*]] = bitcast i64* %p to i8*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[LABEL_ORIGIN:%.*]] = call zeroext i64 @__dfsan_load_label_and_origin(i8* [[INTP]], i64 8)
  ; NO_COMBINE_LOAD_PTR-NEXT: [[LABEL_ORIGIN_H32:%.*]] = lshr i64 [[LABEL_ORIGIN]], 32
  ; NO_COMBINE_LOAD_PTR-NEXT: [[LABEL:%.*]] = trunc i64 [[LABEL_ORIGIN_H32]] to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN:%.*]] = trunc i64 [[LABEL_ORIGIN]] to i32
  ; NO_COMBINE_LOAD_PTR-NEXT: %a = load i64, i64* %p, align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: store i[[#SBITS]] [[LABEL]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: store i32 [[ORIGIN]], i32* @__dfsan_retval_origin_tls, align 4
  
  %a = load i64, i64* %p, align 2
  ret i64 %a
}

define i92 @load92(i92* %p) {
  ; CHECK-LABEL: @"dfs$load92"

  ; NO_COMBINE_LOAD_PTR-LABEL: @"dfs$load92"
  ; NO_COMBINE_LOAD_PTR: [[INTP:%.*]] = ptrtoint i92* %p to i64
  ; NO_COMBINE_LOAD_PTR-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_PTR_0:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_0:%.*]] = load i32, i32* [[ORIGIN_PTR_0]], align 8
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR_0:%.*]] = bitcast i[[#SBITS]]* [[SHADOW_PTR]] to i64*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_0:%.*]] = load i64, i64* [[SHADOW_PTR_0]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR_1:%.*]] = getelementptr i64, i64* [[SHADOW_PTR_0]], i64 1
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_1:%.*]] = load i64, i64* [[SHADOW_PTR_1]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_01:%.*]] = or i64 [[SHADOW_0]], [[SHADOW_1]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_PTR_1:%.*]] = getelementptr i32, i32* [[ORIGIN_PTR_0]], i64 1
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_1:%.*]] = load i32, i32* [[ORIGIN_PTR_1]], align 8
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_PTR_2:%.*]] = getelementptr i64, i64* [[SHADOW_PTR_1]], i64 1
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_2:%.*]] = load i64, i64* [[SHADOW_PTR_2]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64:%.*]] = or i64 [[SHADOW_01]], [[SHADOW_2]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_PTR_2:%.*]] = getelementptr i32, i32* [[ORIGIN_PTR_1]], i64 1
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_2:%.*]] = load i32, i32* [[ORIGIN_PTR_2]], align 8
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_1_NZ:%.*]] = icmp ne i64 [[SHADOW_1]], 0
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN_10:%.*]] = select i1 [[SHADOW_1_NZ]], i32 [[ORIGIN_1]], i32 [[ORIGIN_0]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[SHADOW_2_NZ:%.*]] = icmp ne i64 [[SHADOW_2]], 0
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN:%.*]] = select i1 [[SHADOW_2_NZ]], i32 [[ORIGIN_2]], i32 [[ORIGIN_10]]
  ; NO_COMBINE_LOAD_PTR-NEXT: %a = load i92, i92* %p, align 8
  ; NO_COMBINE_LOAD_PTR-NEXT: store i[[#SBITS]] [[SHADOW]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: store i32 [[ORIGIN]], i32* @__dfsan_retval_origin_tls, align 4
  
  %a = load i92, i92* %p
  ret i92 %a
}

define i17 @load17(i17* %p) {
  ; CHECK-LABEL: @"dfs$load17"

  ; NO_COMBINE_LOAD_PTR-LABEL: @"dfs$load17"
  ; NO_COMBINE_LOAD_PTR-NEXT: [[INTP:%.*]] = bitcast i17* %p to i8*
  ; NO_COMBINE_LOAD_PTR-NEXT: [[LABEL_ORIGIN:%.*]] = call zeroext i64 @__dfsan_load_label_and_origin(i8* [[INTP]], i64 3)
  ; NO_COMBINE_LOAD_PTR-NEXT: [[LABEL_ORIGIN_H32:%.*]] = lshr i64 [[LABEL_ORIGIN]], 32
  ; NO_COMBINE_LOAD_PTR-NEXT: [[LABEL:%.*]] = trunc i64 [[LABEL_ORIGIN_H32]] to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR-NEXT: [[ORIGIN:%.*]] = trunc i64 [[LABEL_ORIGIN]] to i32
  ; NO_COMBINE_LOAD_PTR-NEXT: %a = load i17, i17* %p, align 4
  ; NO_COMBINE_LOAD_PTR-NEXT: store i[[#SBITS]] [[LABEL]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: store i32 [[ORIGIN]], i32* @__dfsan_retval_origin_tls, align 4
  
  %a = load i17, i17* %p, align 4
  ret i17 %a
}

define void @store_zero_to_non_escaped_alloca() {
  ; CHECK-LABEL: @"dfs$store_zero_to_non_escaped_alloca"
  ; CHECK-NEXT: [[A:%.*]] = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK-NEXT: %_dfsa = alloca i32, align 4
  ; CHECK-NEXT: %p = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK-NEXT: store i[[#SBITS]] 0, i[[#SBITS]]* [[A]], align [[#SBYTES]]
  ; CHECK-NEXT: store i16 1, i16* %p, align 2
  ; CHECK-NEXT: ret void
  
  %p = alloca i16
  store i16 1, i16* %p
  ret void
}

define void @store_nonzero_to_non_escaped_alloca(i16 %a) {
  ; CHECK-LABEL: @"dfs$store_nonzero_to_non_escaped_alloca"
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: %_dfsa = alloca i32, align 4
  ; CHECK: store i32 [[AO]], i32* %_dfsa, align 4
  
  %p = alloca i16
  store i16 %a, i16* %p
  ret void
}

declare void @foo(i16* %p)

define void @store_zero_to_escaped_alloca() {
  ; CHECK-LABEL: @"dfs$store_zero_to_escaped_alloca"
  ; CHECK: [[SA:%.*]] = bitcast i[[#SBITS]]* {{.*}} to i32*
  ; CHECK-NEXT: store i32 0, i32* [[SA]], align 2
  ; CHECK-NEXT: store i[[#SBITS]] 1, i[[#SBITS]]* %p, align 2
  ; CHECK-NEXT: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[#SBYTES]]

  ; COMBINE_STORE_PTR-LABEL: @"dfs$store_zero_to_escaped_alloca"
  ; COMBINE_STORE_PTR: [[SA:%.*]] = bitcast i[[#SBITS]]* {{.*}} to i32*
  ; COMBINE_STORE_PTR_NEXT: store i32 0, i32* [[SA]], align 2
  ; COMBINE_STORE_PTR_NEXT: store i16 1, i16* %p, align 2
  ; COMBINE_STORE_PTR_NEXT: call void @foo(i16* %p)
  ; COMBINE_STORE_PTR_NEXT: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[#SBYTES]]

  %p = alloca i16
  store i16 1, i16* %p
  call void @foo(i16* %p)
  ret void
}

define void @store_nonzero_to_escaped_alloca(i16 %a) {
  ; CHECK-LABEL: @"dfs$store_nonzero_to_escaped_alloca"
  ; CHECK-NEXT: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK-NEXT: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
  ; CHECK-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]]
  ; CHECK: [[ORIGIN_OFFSET:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; CHECK-NEXT: [[ORIGIN_ADDR:%.*]] = and i64 [[ORIGIN_OFFSET]], -4
  ; CHECK-NEXT: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; CHECK: %_dfscmp = icmp ne i[[#SBITS]] [[AS]], 0
  ; CHECK-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK: [[L1]]:
  ; CHECK-NEXT: [[NO:%.*]] = call i32 @__dfsan_chain_origin(i32 [[AO]])
  ; CHECK-NEXT: store i32 [[NO]], i32* [[ORIGIN_PTR]], align 4
  ; CHECK-NEXT: br label %[[L2]]
  ; CHECK: [[L2]]:
  ; CHECK-NEXT: store i16 %a, i16* %p, align 2
  
  ; COMBINE_STORE_PTR-LABEL: @"dfs$store_nonzero_to_escaped_alloca"
  ; COMBINE_STORE_PTR-NEXT: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; COMBINE_STORE_PTR-NEXT: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; COMBINE_STORE_PTR: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
  ; COMBINE_STORE_PTR-NEXT: [[OFFSET:%.*]] = and i64 [[INTP]]
  ; COMBINE_STORE_PTR: [[ORIGIN_OFFSET:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; COMBINE_STORE_PTR-NEXT: [[ORIGIN_ADDR:%.*]] = and i64 [[ORIGIN_OFFSET]], -4
  ; COMBINE_STORE_PTR-NEXT: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; COMBINE_STORE_PTR: %_dfscmp = icmp ne i[[#SBITS]] [[AS]], 0
  ; COMBINE_STORE_PTR-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; COMBINE_STORE_PTR: [[L1]]:
  ; COMBINE_STORE_PTR-NEXT: [[NO:%.*]] = call i32 @__dfsan_chain_origin(i32 [[AO]])
  ; COMBINE_STORE_PTR-NEXT: store i32 [[NO]], i32* [[ORIGIN_PTR]], align 4
  ; COMBINE_STORE_PTR-NEXT: br label %[[L2]]
  ; COMBINE_STORE_PTR: [[L2]]:
  ; COMBINE_STORE_PTR-NEXT: store i16 %a, i16* %p, align 2
  
  %p = alloca i16
  store i16 %a, i16* %p
  call void @foo(i16* %p)
  ret void
}

define void @store64_align8(i64* %p, i64 %a) {
  ; CHECK-LABEL: @"dfs$store64_align8"
  ; CHECK-NEXT: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: %_dfscmp = icmp ne i[[#SBITS]] [[AS]], 0
  ; CHECK-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK: [[L1]]:
  ; CHECK-NEXT: [[NO:%.*]] = call i32 @__dfsan_chain_origin(i32 [[AO]])
  ; CHECK-NEXT: [[NO_ZEXT:%.*]] = zext i32 [[NO]] to i64
  ; CHECK-NEXT: [[NO_SHL:%.*]] = shl i64 [[NO_ZEXT]], 32
  ; CHECK-NEXT: [[NO2:%.*]] = or i64 [[NO_ZEXT]], [[NO_SHL]]
  ; CHECK-NEXT: [[O_PTR:%.*]] = bitcast i32* {{.*}} to i64*
  ; CHECK-NEXT: store i64 [[NO2]], i64* [[O_PTR]], align 8
  ; CHECK-NEXT: br label %[[L2]]
  ; CHECK: [[L2]]:
  ; CHECK-NEXT: store i64 %a, i64* %p, align 8
  
  ; COMBINE_STORE_PTR-LABEL: @"dfs$store64_align8"
  ; COMBINE_STORE_PTR-NEXT: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; COMBINE_STORE_PTR-NEXT: [[PS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; COMBINE_STORE_PTR-NEXT: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; COMBINE_STORE_PTR-NEXT: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[#SBYTES]]
  ; COMBINE_STORE_PTR-NEXT: [[MS:%.*]] = or i[[#SBITS]] [[AS]], [[PS]]
  ; COMBINE_STORE_PTR-NEXT: [[NE:%.*]] = icmp ne i[[#SBITS]] [[PS]], 0
  ; COMBINE_STORE_PTR-NEXT: [[MO:%.*]] = select i1 [[NE]], i32 [[PO]], i32 [[AO]]
  ; COMBINE_STORE_PTR: %_dfscmp = icmp ne i[[#SBITS]] [[MS]], 0
  ; COMBINE_STORE_PTR-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; COMBINE_STORE_PTR: [[L1]]:
  ; COMBINE_STORE_PTR-NEXT: [[NO:%.*]] = call i32 @__dfsan_chain_origin(i32 [[MO]])
  ; COMBINE_STORE_PTR-NEXT: [[NO_ZEXT:%.*]] = zext i32 [[NO]] to i64
  ; COMBINE_STORE_PTR-NEXT: [[NO_SHL:%.*]] = shl i64 [[NO_ZEXT]], 32
  ; COMBINE_STORE_PTR-NEXT: [[NO2:%.*]] = or i64 [[NO_ZEXT]], [[NO_SHL]]
  ; COMBINE_STORE_PTR-NEXT: [[O_PTR:%.*]] = bitcast i32* {{.*}} to i64*
  ; COMBINE_STORE_PTR-NEXT: store i64 [[NO2]], i64* [[O_PTR]], align 8
  ; COMBINE_STORE_PTR-NEXT: br label %[[L2]]
  ; COMBINE_STORE_PTR: [[L2]]:
  ; COMBINE_STORE_PTR-NEXT: store i64 %a, i64* %p, align 8
  
  store i64 %a, i64* %p
  ret void
}

define void @store64_align2(i64* %p, i64 %a) {
  ; CHECK-LABEL: @"dfs$store64_align2"
  ; CHECK-NEXT: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: %_dfscmp = icmp ne i[[#SBITS]] [[AS]], 0
  ; CHECK-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK: [[L1]]:
  ; CHECK-NEXT: [[NO:%.*]] = call i32 @__dfsan_chain_origin(i32 [[AO]])
  ; CHECK-NEXT: store i32 [[NO]], i32* [[O_PTR0:%.*]], align 4
  ; CHECK-NEXT: [[O_PTR1:%.*]] = getelementptr i32, i32* [[O_PTR0]], i32 1
  ; CHECK-NEXT: store i32 [[NO]], i32* [[O_PTR1]], align 4
  ; CHECK: [[L2]]:
  ; CHECK-NEXT: store i64 %a, i64* %p, align [[#SBYTES]]
  
  store i64 %a, i64* %p, align 2
  ret void
}

define void @store96_align8(i96* %p, i96 %a) {
  ; CHECK-LABEL: @"dfs$store96_align8"
  ; CHECK-NEXT: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK-NEXT: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: %_dfscmp = icmp ne i[[#SBITS]] [[AS]], 0
  ; CHECK-NEXT: br i1 %_dfscmp, label %[[L1:.*]], label %[[L2:.*]],
  ; CHECK: [[L1]]:
  ; CHECK-NEXT: [[NO:%.*]] = call i32 @__dfsan_chain_origin(i32 [[AO]])
  ; CHECK-NEXT: [[NO_ZEXT:%.*]] = zext i32 [[NO]] to i64
  ; CHECK-NEXT: [[NO_SHL:%.*]] = shl i64 [[NO_ZEXT]], 32
  ; CHECK-NEXT: [[NO2:%.*]] = or i64 [[NO_ZEXT]], [[NO_SHL]]
  ; CHECK-NEXT: [[O_PTR64:%.*]] = bitcast i32* [[O_PTR0:%.*]] to i64*
  ; CHECK-NEXT: store i64 [[NO2]], i64* [[O_PTR64]], align 8
  ; CHECK-NEXT: [[O_PTR1:%.*]] = getelementptr i32, i32* [[O_PTR0]], i32 2
  ; CHECK-NEXT: store i32 [[NO]], i32* [[O_PTR1]], align 8
  ; CHECK: [[L2]]:
  ; CHECK-NEXT: store i96 %a, i96* %p, align 8
  
  store i96 %a, i96* %p, align 8
  ret void
}
