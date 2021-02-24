; RUN: opt < %s -dfsan -dfsan-track-origins=1 -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefix=CHECK
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define float @unop(float %f) {
  ; CHECK: @"dfs$unop"
  ; CHECK: [[FO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: store i32 [[FO]], i32* @__dfsan_retval_origin_tls, align 4

  %r = fneg float %f
  ret float %r
}

define i1 @binop(i1 %a, i1 %b) {
  ; CHECK: @"dfs$binop"
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[BS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align 2
  ; CHECK: [[NE:%.*]] = icmp ne i16 [[BS]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], i32* @__dfsan_retval_origin_tls, align 4

  %r = add i1 %a, %b
  ret i1 %r
}

define i8 @castop(i32* %p) {
  ; CHECK: @"dfs$castop"
  ; CHECK: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: store i32 [[PO]], i32* @__dfsan_retval_origin_tls, align 4

  %r = ptrtoint i32* %p to i8
  ret i8 %r
}

define i1 @cmpop(i1 %a, i1 %b) {
  ; CHECK: @"dfs$cmpop"
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[BS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align 2
  ; CHECK: [[NE:%.*]] = icmp ne i16 [[BS]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], i32* @__dfsan_retval_origin_tls, align 4

  %r = icmp eq i1 %a, %b
  ret i1 %r
}

define i32* @gepop([10 x [20 x i32]]* %p, i32 %a, i32 %b, i32 %c) {
  ; CHECK: @"dfs$gepop"
  ; CHECK: [[CO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 3), align 4
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 2), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[CS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 6) to i16*), align 2
  ; CHECK: [[BS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to i16*), align 2
  ; CHECK: [[AS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align 2
  ; CHECK: [[AS_NE:%.*]] = icmp ne i16 [[AS]], 0
  ; CHECK: [[APO:%.*]] = select i1 [[AS_NE]], i32 [[AO]], i32 [[PO]]
  ; CHECK: [[BS_NE:%.*]] = icmp ne i16 [[BS]], 0
  ; CHECK: [[ABPO:%.*]] = select i1 [[BS_NE]], i32 [[BO]], i32 [[APO]]
  ; CHECK: [[CS_NE:%.*]] = icmp ne i16 [[CS]], 0
  ; CHECK: [[ABCPO:%.*]] = select i1 [[CS_NE]], i32 [[CO]], i32 [[ABPO]]
  ; CHECK: store i32 [[ABCPO]], i32* @__dfsan_retval_origin_tls, align 4

  %e = getelementptr [10 x [20 x i32]], [10 x [20 x i32]]* %p, i32 %a, i32 %b, i32 %c
  ret i32* %e
}

define i32 @eeop(<4 x i32> %a, i32 %b) {
  ; CHECK: @"dfs$eeop"
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[BS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align 2
  ; CHECK: [[NE:%.*]] = icmp ne i16 [[BS]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], i32* @__dfsan_retval_origin_tls, align 4

  %e = extractelement <4 x i32> %a, i32 %b
  ret i32 %e
}

define <4 x i32> @ieop(<4 x i32> %p, i32 %a, i32 %b) {
  ; CHECK: @"dfs$ieop"
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 2), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[PO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[BS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to i16*), align 2
  ; CHECK: [[AS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align 2
  ; CHECK: [[AS_NE:%.*]] = icmp ne i16 [[AS]], 0
  ; CHECK: [[APO:%.*]] = select i1 [[AS_NE]], i32 [[AO]], i32 [[PO]]
  ; CHECK: [[BS_NE:%.*]] = icmp ne i16 [[BS]], 0
  ; CHECK: [[ABPO:%.*]] = select i1 [[BS_NE]], i32 [[BO]], i32 [[APO]]
  ; CHECK: store i32 [[ABPO]], i32* @__dfsan_retval_origin_tls, align 4

  %e = insertelement <4 x i32> %p, i32 %a, i32 %b
  ret <4 x i32> %e
}

define <4 x i32> @svop(<4 x i32> %a, <4 x i32> %b) {
  ; CHECK: @"dfs$svop"
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[BS:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align 2
  ; CHECK: [[NE:%.*]] = icmp ne i16 [[BS]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], i32* @__dfsan_retval_origin_tls, align 4
  
  %e = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %e
}  

define i32 @evop({i32, float} %a) {
  ; CHECK: @"dfs$evop"
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: store i32 [[AO]], i32* @__dfsan_retval_origin_tls, align 4

  %e = extractvalue {i32, float} %a, 0
  ret i32 %e
}

define {i32, {float, float}} @ivop({i32, {float, float}} %a, {float, float} %b) {
  ; CHECK: @"dfs$ivop"
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[BS:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 6) to { i16, i16 }*), align 2
  ; CHECK: [[BS0:%.*]] = extractvalue { i16, i16 } [[BS]], 0
  ; CHECK: [[BS1:%.*]] = extractvalue { i16, i16 } [[BS]], 1
  ; CHECK: [[BS01:%.*]] = or i16 [[BS0]], [[BS1]]
  ; CHECK: [[NE:%.*]] = icmp ne i16 [[BS01]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], i32* @__dfsan_retval_origin_tls, align 4
  
  %e = insertvalue {i32, {float, float}} %a, {float, float} %b, 1
  ret {i32, {float, float}} %e
}