; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1  -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
define float @unop(float %f) {
  ; CHECK: @unop.dfsan
  ; CHECK: [[FO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: store i32 [[FO]], ptr @__dfsan_retval_origin_tls, align 4

  %r = fneg float %f
  ret float %r
}

define i1 @binop(i1 %a, i1 %b) {
  ; CHECK: @binop.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[NE:%.*]] = icmp ne i8 [[BS]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], ptr @__dfsan_retval_origin_tls, align 4

  %r = add i1 %a, %b
  ret i1 %r
}

define i8 @castop(ptr %p) {
  ; CHECK: @castop.dfsan
  ; CHECK: [[PO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: store i32 [[PO]], ptr @__dfsan_retval_origin_tls, align 4

  %r = ptrtoint ptr %p to i8
  ret i8 %r
}

define i1 @cmpop(i1 %a, i1 %b) {
  ; CHECK: @cmpop.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[NE:%.*]] = icmp ne i8 [[BS]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], ptr @__dfsan_retval_origin_tls, align 4

  %r = icmp eq i1 %a, %b
  ret i1 %r
}

define ptr @gepop(ptr %p, i32 %a, i32 %b, i32 %c) {
  ; CHECK: @gepop.dfsan
  ; CHECK: [[CO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 3), align 4
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 2), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[PO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[CS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 6) to ptr), align 2
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align 2
  ; CHECK: [[AS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS_NE:%.*]] = icmp ne i8 [[AS]], 0
  ; CHECK: [[APO:%.*]] = select i1 [[AS_NE]], i32 [[AO]], i32 [[PO]]
  ; CHECK: [[BS_NE:%.*]] = icmp ne i8 [[BS]], 0
  ; CHECK: [[ABPO:%.*]] = select i1 [[BS_NE]], i32 [[BO]], i32 [[APO]]
  ; CHECK: [[CS_NE:%.*]] = icmp ne i8 [[CS]], 0
  ; CHECK: [[ABCPO:%.*]] = select i1 [[CS_NE]], i32 [[CO]], i32 [[ABPO]]
  ; CHECK: store i32 [[ABCPO]], ptr @__dfsan_retval_origin_tls, align 4

  %e = getelementptr [10 x [20 x i32]], ptr %p, i32 %a, i32 %b, i32 %c
  ret ptr %e
}

define i32 @eeop(<4 x i32> %a, i32 %b) {
  ; CHECK: @eeop.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[NE:%.*]] = icmp ne i8 [[BS]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], ptr @__dfsan_retval_origin_tls, align 4

  %e = extractelement <4 x i32> %a, i32 %b
  ret i32 %e
}

define <4 x i32> @ieop(<4 x i32> %p, i32 %a, i32 %b) {
  ; CHECK: @ieop.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 2), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[PO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align 2
  ; CHECK: [[AS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS_NE:%.*]] = icmp ne i8 [[AS]], 0
  ; CHECK: [[APO:%.*]] = select i1 [[AS_NE]], i32 [[AO]], i32 [[PO]]
  ; CHECK: [[BS_NE:%.*]] = icmp ne i8 [[BS]], 0
  ; CHECK: [[ABPO:%.*]] = select i1 [[BS_NE]], i32 [[BO]], i32 [[APO]]
  ; CHECK: store i32 [[ABPO]], ptr @__dfsan_retval_origin_tls, align 4

  %e = insertelement <4 x i32> %p, i32 %a, i32 %b
  ret <4 x i32> %e
}

define <4 x i32> @svop(<4 x i32> %a, <4 x i32> %b) {
  ; CHECK: @svop.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[NE:%.*]] = icmp ne i8 [[BS]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], ptr @__dfsan_retval_origin_tls, align 4
  
  %e = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 1, i32 5>
  ret <4 x i32> %e
}  

define i32 @evop({i32, float} %a) {
  ; CHECK: @evop.dfsan
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: store i32 [[AO]], ptr @__dfsan_retval_origin_tls, align 4

  %e = extractvalue {i32, float} %a, 0
  ret i32 %e
}

define {i32, {float, float}} @ivop({i32, {float, float}} %a, {float, float} %b) {
  ; CHECK: @ivop.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; COMM: TODO simplify the expression 4 to
  ; COMM: 6, if shadow-tls-alignment is updated to match shadow
  ; CHECK: [[BS:%.*]] = load { i8, i8 }, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align 2
  ; CHECK: [[BS0:%.*]] = extractvalue { i8, i8 } [[BS]], 0
  ; CHECK: [[BS1:%.*]] = extractvalue { i8, i8 } [[BS]], 1
  ; CHECK: [[BS01:%.*]] = or i8 [[BS0]], [[BS1]]
  ; CHECK: [[NE:%.*]] = icmp ne i8 [[BS01]], 0
  ; CHECK: [[MO:%.*]] = select i1 [[NE]], i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[MO]], ptr @__dfsan_retval_origin_tls, align 4
  
  %e = insertvalue {i32, {float, float}} %a, {float, float} %b, 1
  ret {i32, {float, float}} %e
}
