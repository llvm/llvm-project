; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefixes=CHECK_META,CHECK
; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s --check-prefixes=CHECK_META,NO_COMBINE_LOAD_PTR
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK_META: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK_META: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define {} @load0({}* %p) {
  ; CHECK: @"dfs$load0"
  ; CHECK-NEXT: %a = load {}, {}* %p, align 1
  ; CHECK-NEXT: store {} zeroinitializer, {}* bitcast ([100 x i64]* @__dfsan_retval_tls to {}*), align [[#SBYTES]]
  ; CHECK-NEXT: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK-NEXT: ret {} %a

  %a = load {}, {}* %p
  ret {} %a
}

define i16 @load_non_escaped_alloca() {
  ; CHECK: @"dfs$load_non_escaped_alloca"
  ; CHECK: [[S_ALLOCA:%.*]] = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK: [[O_ALLOCA:%.*]] = alloca i32, align 4
  ; CHECK: [[SHADOW:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[S_ALLOCA]], align [[#SBYTES]]
  ; CHECK: [[ORIGIN:%.*]] = load i32, i32* [[O_ALLOCA]], align 4
  ; CHECK: %a = load i16, i16* %p, align 2
  ; CHECK: store i[[#SBITS]] [[SHADOW]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: store i32 [[ORIGIN]], i32* @__dfsan_retval_origin_tls, align 4
  
  %p = alloca i16
  %a = load i16, i16* %p
  ret i16 %a
}

define i16* @load_escaped_alloca() {
  ; CHECK: @"dfs$load_escaped_alloca"
  ; CHECK: [[INTP:%.*]] = ptrtoint i[[#SBITS]]* %p to i64
  ; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
  ; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; CHECK: [[SHADOW_PTR0:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; CHECK: [[ORIGIN_OFFSET:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; CHECK: [[ORIGIN_ADDR:%.*]] = and i64 [[ORIGIN_OFFSET]], -4
  ; CHECK: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; CHECK: {{%.*}} = load i32, i32* [[ORIGIN_PTR]], align 4
  ; CHECK: [[SHADOW_PTR1:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR0]], i64 1
  ; CHECK: [[SHADOW0:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR0]], align [[#SBYTES]]
  ; CHECK: [[SHADOW1:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR1]], align [[#SBYTES]]
  ; CHECK: {{%.*}} = or i[[#SBITS]] [[SHADOW0]], [[SHADOW1]]
  ; CHECK: %a = load i16, i16* %p, align 2
  ; CHECK: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  
  %p = alloca i16
  %a = load i16, i16* %p
  ret i16* %p
}

@X = constant i1 1
define i1 @load_global() {
  ; CHECK: @"dfs$load_global"
  ; CHECK: %a = load i1, i1* @X, align 1
  ; CHECK: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: store i32 0, i32* @__dfsan_retval_origin_tls, align 4

  %a = load i1, i1* @X
  ret i1 %a
}

define i1 @load1(i1* %p) {
  ; CHECK: @"dfs$load1"
  ; CHECK: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[PS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
  ; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
  ; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; CHECK: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; CHECK: [[ORIGIN_OFFSET:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; CHECK: [[ORIGIN_ADDR:%.*]] = and i64 [[ORIGIN_OFFSET]], -4
  ; CHECK: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; CHECK: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 4
  ; CHECK: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR]], align [[#SBYTES]]
  ; CHECK: [[RS:%.*]] = or i[[#SBITS]] [[AS]], [[PS]]
  ; CHECK: [[PS_NZ:%.*]] = icmp ne i[[#SBITS]] [[PS]], 0
  ; CHECK: [[RO:%.*]] = select i1 [[PS_NZ]], i32 [[PO]], i32 [[AO]]
  ; CHECK: %a = load i1, i1* %p, align 1
  ; CHECK: store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i1, i1* %p
  ret i1 %a
}

define i16 @load16(i1 %i, i16* %p) {
  ; CHECK: @"dfs$load16"
  ; CHECK: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[PS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: [[INTP:%.*]] = ptrtoint {{.*}} %p to i64
  ; CHECK: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
  ; CHECK: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; CHECK: [[SHADOW_PTR0:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; CHECK: [[ORIGIN_OFFSET:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; CHECK: [[ORIGIN_ADDR:%.*]] = and i64 [[ORIGIN_OFFSET]], -4
  ; CHECK: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; CHECK: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 4
  ; CHECK: [[SHADOW_PTR1:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR0]], i64 1
  ; CHECK: [[SHADOW0:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR0]], align [[#SBYTES]]
  ; CHECK: [[SHADOW1:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[SHADOW_PTR1]], align [[#SBYTES]]
  ; CHECK: [[AS:%.*]] = or i[[#SBITS]] [[SHADOW0]], [[SHADOW1]]
  ; CHECK: [[RS:%.*]] = or i[[#SBITS]] [[AS]], [[PS]]
  ; CHECK: [[PS_NZ:%.*]] = icmp ne i[[#SBITS]] [[PS]], 0
  ; CHECK: [[RO:%.*]] = select i1 [[PS_NZ]], i32 [[PO]], i32 [[AO]]
  ; CHECK: %a = load i16, i16* %p, align 2
  ; CHECK: store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; CHECK: store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i16, i16* %p
  ret i16 %a
}

define i32 @load32(i32* %p) {
  ; CHECK: @"dfs$load32"

  ; NO_COMBINE_LOAD_PTR: @"dfs$load32"
  ; NO_COMBINE_LOAD_PTR: [[INTP:%.*]] = ptrtoint i32* %p to i64
  ; NO_COMBINE_LOAD_PTR: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_PTR:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; NO_COMBINE_LOAD_PTR: [[AO:%.*]] = load i32, i32* [[ORIGIN_PTR]], align 4
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR64:%.*]] = bitcast i[[#SBITS]]* [[SHADOW_PTR]] to i64*
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64:%.*]] = load i64, i64* [[SHADOW_PTR64]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR: %a = load i32, i32* %p, align 4
  ; NO_COMBINE_LOAD_PTR: store i[[#SBITS]] [[SHADOW]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: store i32 [[AO]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i32, i32* %p
  ret i32 %a
}

define i64 @load64(i64* %p) {
  ; CHECK: @"dfs$load64"
  
  ; NO_COMBINE_LOAD_PTR: @"dfs$load64"
  ; NO_COMBINE_LOAD_PTR: [[INTP:%.*]] = ptrtoint i64* %p to i64
  ; NO_COMBINE_LOAD_PTR: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_PTR_0:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_0:%.*]] = load i32, i32* [[ORIGIN_PTR_0]], align 8
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR_0:%.*]] = bitcast i[[#SBITS]]* [[SHADOW_PTR]] to i64*
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_0:%.*]] = load i64, i64* [[SHADOW_PTR_0]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR_1:%.*]] = getelementptr i64, i64* [[SHADOW_PTR_0]], i64 1
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_1:%.*]] = load i64, i64* [[SHADOW_PTR_1]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64:%.*]] = or i64 [[SHADOW_0]], [[SHADOW_1]]
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_PTR_1:%.*]] = getelementptr i32, i32* [[ORIGIN_PTR_0]], i64 1
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_1:%.*]] = load i32, i32* [[ORIGIN_PTR_1]], align 8
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_1_NZ:%.*]] = icmp ne i64 [[SHADOW_1]], 0
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN:%.*]] = select i1 [[SHADOW_1_NZ]], i32 [[ORIGIN_1]], i32 [[ORIGIN_0]]
  ; NO_COMBINE_LOAD_PTR: %a = load i64, i64* %p, align 8
  ; NO_COMBINE_LOAD_PTR: store i[[#SBITS]] [[SHADOW]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: store i32 [[ORIGIN]], i32* @__dfsan_retval_origin_tls, align 4

  %a = load i64, i64* %p
  ret i64 %a
}

define i64 @load64_align2(i64* %p) {
  ; CHECK: @"dfs$load64_align2"

  ; NO_COMBINE_LOAD_PTR: @"dfs$load64_align2"
  ; NO_COMBINE_LOAD_PTR-NEXT: [[INTP:%.*]] = bitcast i64* %p to i8*
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
  ; CHECK: @"dfs$load92"

  ; NO_COMBINE_LOAD_PTR: @"dfs$load92"
  ; NO_COMBINE_LOAD_PTR: [[INTP:%.*]] = ptrtoint i92* %p to i64
  ; NO_COMBINE_LOAD_PTR: [[OFFSET:%.*]] = and i64 [[INTP]], -123145302310913
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_ADDR:%.*]] = mul i64 [[OFFSET]], 2
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR:%.*]] = inttoptr i64 [[SHADOW_ADDR]] to i[[#SBITS]]*
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_ADDR:%.*]] = add i64 [[OFFSET]], 35184372088832
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_PTR_0:%.*]] = inttoptr i64 [[ORIGIN_ADDR]] to i32*
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_0:%.*]] = load i32, i32* [[ORIGIN_PTR_0]], align 8
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR_0:%.*]] = bitcast i[[#SBITS]]* [[SHADOW_PTR]] to i64*
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_0:%.*]] = load i64, i64* [[SHADOW_PTR_0]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR_1:%.*]] = getelementptr i64, i64* [[SHADOW_PTR_0]], i64 1
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_1:%.*]] = load i64, i64* [[SHADOW_PTR_1]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_01:%.*]] = or i64 [[SHADOW_0]], [[SHADOW_1]]
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_PTR_1:%.*]] = getelementptr i32, i32* [[ORIGIN_PTR_0]], i64 1
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_1:%.*]] = load i32, i32* [[ORIGIN_PTR_1]], align 8
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_PTR_2:%.*]] = getelementptr i64, i64* [[SHADOW_PTR_1]], i64 1
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_2:%.*]] = load i64, i64* [[SHADOW_PTR_2]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64:%.*]] = or i64 [[SHADOW_01]], [[SHADOW_2]]
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_PTR_2:%.*]] = getelementptr i32, i32* [[ORIGIN_PTR_1]], i64 1
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_2:%.*]] = load i32, i32* [[ORIGIN_PTR_2]], align 8
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_H32:%.*]] = lshr i64 [[SHADOW64]], 32
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32:%.*]] = or i64 [[SHADOW64]], [[SHADOW64_H32]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32_H16:%.*]] = lshr i64 [[SHADOW64_HL32]], 16
  ; NO_COMBINE_LOAD_PTR: [[SHADOW64_HL32_HL16:%.*]] = or i64 [[SHADOW64_HL32]], [[SHADOW64_HL32_H16]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW:%.*]] = trunc i64 [[SHADOW64_HL32_HL16]] to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_1_NZ:%.*]] = icmp ne i64 [[SHADOW_1]], 0
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN_10:%.*]] = select i1 [[SHADOW_1_NZ]], i32 [[ORIGIN_1]], i32 [[ORIGIN_0]]
  ; NO_COMBINE_LOAD_PTR: [[SHADOW_2_NZ:%.*]] = icmp ne i64 [[SHADOW_2]], 0
  ; NO_COMBINE_LOAD_PTR: [[ORIGIN:%.*]] = select i1 [[SHADOW_2_NZ]], i32 [[ORIGIN_2]], i32 [[ORIGIN_10]]
  ; NO_COMBINE_LOAD_PTR: %a = load i92, i92* %p, align 8
  ; NO_COMBINE_LOAD_PTR: store i[[#SBITS]] [[SHADOW]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: store i32 [[ORIGIN]], i32* @__dfsan_retval_origin_tls, align 4
  
  %a = load i92, i92* %p
  ret i92 %a
}

define i17 @load17(i17* %p) {
  ; CHECK: @"dfs$load17"

  ; NO_COMBINE_LOAD_PTR: @"dfs$load17"
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
