; RUN: opt < %s -dfsan -dfsan-track-origins=1  -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define i32 @discard(i32 %a, i32 %b) {
  ret i32 0
}

define i32 @call_discard(i32 %a, i32 %b) {
  ; CHECK: @call_discard.dfsan
  ; CHECK: %r = call i32 @discard(i32 %a, i32 %b)
  ; CHECK: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK: ret i32 %r

  %r = call i32 @discard(i32 %a, i32 %b)
  ret i32 %r
}

; CHECK: i32 @functional(i32 %a, i32 %b)
define i32 @functional(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @call_functional(i32 %a, i32 %b) {
  ; CHECK: @call_functional.dfsan
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[RO:%.*]] = select i1 {{.*}}, i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4

  %r = call i32 @functional(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @uninstrumented(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @call_uninstrumented(i32 %a, i32 %b) {
  ; CHECK: @call_uninstrumented.dfsan
  ; CHECK: %r = call i32 @uninstrumented(i32 %a, i32 %b)
  ; CHECK: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK: ret i32 %r

  %r = call i32 @uninstrumented(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @g(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

@discardg = alias i32 (i32, i32), i32 (i32, i32)* @g

define i32 @call_discardg(i32 %a, i32 %b) {
  ; CHECK: @call_discardg.dfsan
  ; CHECK: %r = call i32 @discardg(i32 %a, i32 %b)
  ; CHECK: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  ; CHECK: ret i32 %r

  %r = call i32 @discardg(i32 %a, i32 %b)
  ret i32 %r
}

define void @custom_without_ret(i32 %a, i32 %b) {
  ret void
}

define i32 @custom_with_ret(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

define void @custom_varg_without_ret(i32 %a, i32 %b, ...) {
  ret void
}

define i32 @custom_varg_with_ret(i32 %a, i32 %b, ...) {
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @custom_cb_with_ret(i32 (i32, i32)* %cb, i32 %a, i32 %b) {
  %r = call i32 %cb(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @cb_with_ret(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

define void @custom_cb_without_ret(void (i32, i32)* %cb, i32 %a, i32 %b) {
  call void %cb(i32 %a, i32 %b)
  ret void
}

define void @cb_without_ret(i32 %a, i32 %b) {
  ret void
}

define i32 (i32, i32)* @ret_custom() {
  ; CHECK: @ret_custom.dfsan
  ; CHECK: store i32 0, i32* @__dfsan_retval_origin_tls, align 4
  
  ret i32 (i32, i32)* @custom_with_ret
}

define void @call_custom_without_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_without_ret.dfsan
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
  ; CHECK: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK: call void @__dfso_custom_without_ret(i32 %a, i32 %b, i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i32 zeroext [[AO]], i32 zeroext [[BO]])
  ; CHECK-NEXT: ret void

  call void @custom_without_ret(i32 %a, i32 %b)
  ret void
}

define i32 @call_custom_with_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_with_ret.dfsan
  ; CHECK: %originreturn = alloca i32, align 4
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK: [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
  ; CHECK: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK: {{.*}} = call i32 @__dfso_custom_with_ret(i32 %a, i32 %b, i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i[[#SBITS]]* %labelreturn, i32 zeroext [[AO]], i32 zeroext [[BO]], i32* %originreturn)
  ; CHECK: [[RS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; CHECK: [[RO:%.*]] = load i32, i32* %originreturn, align 4
  ; CHECK: store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK: store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4

  %r = call i32 @custom_with_ret(i32 %a, i32 %b)
  ret i32 %r
}

define void @call_custom_varg_without_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_varg_without_ret.dfsan
  ; CHECK: %originva = alloca [1 x i32], align 4
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: %labelva = alloca [1 x i[[#SBITS]]], align [[#SBYTES]]
  ; CHECK: [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
  ; CHECK: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK: [[VS0:%.*]] = getelementptr inbounds [1 x i[[#SBITS]]], [1 x i[[#SBITS]]]* %labelva, i32 0, i32 0
  ; CHECK: store i[[#SBITS]] [[AS]], i[[#SBITS]]* [[VS0]], align [[#SBYTES]]
  ; CHECK: [[VS0:%.*]] = getelementptr inbounds [1 x i[[#SBITS]]], [1 x i[[#SBITS]]]* %labelva, i32 0, i32 0
  ; CHECK: [[VO0:%.*]] = getelementptr inbounds [1 x i32], [1 x i32]* %originva, i32 0, i32 0
  ; CHECK: store i32 [[AO]], i32* [[VO0]], align 4
  ; CHECK: [[VO0:%.*]] = getelementptr inbounds [1 x i32], [1 x i32]* %originva, i32 0, i32 0
  ; CHECK: call void (i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i32, i32, i32*, ...) @__dfso_custom_varg_without_ret(i32 %a, i32 %b, i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i[[#SBITS]]* [[VS0]], i32 zeroext [[AO]], i32 zeroext [[BO]], i32* [[VO0]], i32 %a)
  ; CHECK-NEXT: ret void

  call void (i32, i32, ...) @custom_varg_without_ret(i32 %a, i32 %b, i32 %a)
  ret void
}

define i32 @call_custom_varg_with_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_varg_with_ret.dfsan
  ; CHECK: %originreturn = alloca i32, align 4
  ; CHECK: %originva = alloca [1 x i32], align 4
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK: %labelva = alloca [1 x i[[#SBITS]]], align [[#SBYTES]]
  ; CHECK: [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
  ; CHECK: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK: [[VS0:%.*]] = getelementptr inbounds [1 x i[[#SBITS]]], [1 x i[[#SBITS]]]* %labelva, i32 0, i32 0
  ; CHECK: store i[[#SBITS]] [[BS]], i[[#SBITS]]* [[VS0]], align [[#SBYTES]]
  ; CHECK: [[VS0:%.*]] = getelementptr inbounds [1 x i[[#SBITS]]], [1 x i[[#SBITS]]]* %labelva, i32 0, i32 0
  ; CHECK: [[VO0:%.*]] = getelementptr inbounds [1 x i32], [1 x i32]* %originva, i32 0, i32 0
  ; CHECK: store i32 [[BO]], i32* [[VO0]], align 4
  ; CHECK: [[VO0:%.*]] = getelementptr inbounds [1 x i32], [1 x i32]* %originva, i32 0, i32 0
  ; CHECK: {{.*}} = call i32 (i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i[[#SBITS]]*, i32, i32, i32*, i32*, ...) @__dfso_custom_varg_with_ret(i32 %a, i32 %b, i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i[[#SBITS]]* [[VS0]], i[[#SBITS]]* %labelreturn, i32 zeroext [[AO]], i32 zeroext [[BO]], i32* [[VO0]], i32* %originreturn, i32 %b)
  ; CHECK: [[RS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; CHECK: [[RO:%.*]] = load i32, i32* %originreturn, align 4
  ; CHECK: store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK: store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4

  %r = call i32 (i32, i32, ...) @custom_varg_with_ret(i32 %a, i32 %b, i32 %b)
  ret i32 %r
}

define i32 @call_custom_cb_with_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_cb_with_ret.dfsan
  ; CHECK: %originreturn = alloca i32, align 4
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK: [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
  ; CHECK: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK: {{.*}} = call i32 @__dfso_custom_cb_with_ret(i32 (i32 (i32, i32)*, i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i32, i32, i32*)* @"dfst0$custom_cb_with_ret", i8* bitcast (i32 (i32, i32)* @cb_with_ret.dfsan to i8*), i32 %a, i32 %b, i[[#SBITS]] zeroext 0, i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i[[#SBITS]]* %labelreturn, i32 zeroext 0, i32 zeroext [[AO]], i32 zeroext [[BO]], i32* %originreturn)
  ; CHECK: [[RS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; CHECK: [[RO:%.*]] = load i32, i32* %originreturn, align 4
  ; CHECK: store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK: store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4

  %r = call i32 @custom_cb_with_ret(i32 (i32, i32)* @cb_with_ret, i32 %a, i32 %b)
  ret i32 %r
}

define void @call_custom_cb_without_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_cb_without_ret.dfsan
  ; CHECK: [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
  ; CHECK: [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
  ; CHECK: [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
  ; CHECK: call void @__dfso_custom_cb_without_ret(void (void (i32, i32)*, i32, i32, i[[#SBITS]], i[[#SBITS]], i32, i32)* @"dfst0$custom_cb_without_ret", i8* bitcast (void (i32, i32)* @cb_without_ret.dfsan to i8*), i32 %a, i32 %b, i[[#SBITS]] zeroext 0, i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i32 zeroext 0, i32 zeroext [[AO]], i32 zeroext [[BO]])
  ; CHECK-NEXT: ret void

  call void @custom_cb_without_ret(void (i32, i32)* @cb_without_ret, i32 %a, i32 %b)
  ret void
}

; CHECK: define i32 @discardg(i32 %0, i32 %1)
; CHECK: [[R:%.*]] = call i32 @g.dfsan
; CHECK-NEXT: %_dfsret = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT: %_dfsret_o = load i32, i32* @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT: ret i32 [[R]]

; CHECK: define linkonce_odr void @"dfso$custom_without_ret"(i32 %0, i32 %1)
; CHECK:  [[BO:%.*]]  = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  [[AO:%.*]]  = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK-NEXT:  [[BS:%.*]]  = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[AS:%.*]]  = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  call void @__dfso_custom_without_ret(i32 %0, i32 %1, i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i32 zeroext [[AO]], i32 zeroext [[BO]])
; CHECK-NEXT:  ret void

; CHECK: define linkonce_odr i32 @"dfso$custom_with_ret"(i32 %0, i32 %1)
; CHECK:  %originreturn = alloca i32, align 4
; CHECK-NEXT:  [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK-NEXT:  %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
; CHECK-NEXT:  [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[R:%.*]] = call i32 @__dfso_custom_with_ret(i32 %0, i32 %1, i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i[[#SBITS]]* %labelreturn, i32 zeroext [[AO]], i32 zeroext [[BO]], i32* %originreturn)
; CHECK-NEXT:  [[RS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
; CHECK-NEXT:  [[RO:%.*]] = load i32, i32* %originreturn, align 4
; CHECK-NEXT:  store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT:  ret i32 [[R]]

; CHECK: define linkonce_odr void @"dfso$custom_varg_without_ret"(i32 %0, i32 %1, ...)
; CHECK:  call void @__dfsan_vararg_wrapper(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @0, i32 0, i32 0))
; CHECK-NEXT:  unreachable

; CHECK: define linkonce_odr i32 @"dfso$custom_varg_with_ret"(i32 %0, i32 %1, ...)
; CHECK:  call void @__dfsan_vararg_wrapper(i8* getelementptr inbounds ([21 x i8], [21 x i8]* @1, i32 0, i32 0))
; CHECK-NEXT:  unreachable

; CHECK: define linkonce_odr i32 @"dfso$custom_cb_with_ret"(i32 (i32, i32)* %0, i32 %1, i32 %2)
; CHECK:  %originreturn = alloca i32, align 4
; CHECK-NEXT:  [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 2), align 4
; CHECK-NEXT:  [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  [[CO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK-NEXT:  %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
; CHECK-NEXT:  [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[CS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[C:%.*]] = bitcast i32 (i32, i32)* %0 to i8*
; CHECK-NEXT:  [[R:%.*]] = call i32 @__dfso_custom_cb_with_ret(i32 (i32 (i32, i32)*, i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i32, i32, i32*)* @"dfst0$custom_cb_with_ret", i8* [[C]], i32 %1, i32 %2, i[[#SBITS]] zeroext [[CS]], i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i[[#SBITS]]* %labelreturn, i32 zeroext [[CO]], i32 zeroext [[AO]], i32 zeroext [[BO]], i32* %originreturn)
; CHECK-NEXT:  [[RS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
; CHECK-NEXT:  [[RO:%.*]] = load i32, i32* %originreturn, align 4
; CHECK-NEXT:  store i[[#SBITS]] [[RS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  store i32 [[RO]], i32* @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT:  ret i32 [[R]]

; CHECK: define linkonce_odr void @"dfso$custom_cb_without_ret"(void (i32, i32)* %0, i32 %1, i32 %2)
; CHECK:   [[BO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 2), align 4
; CHECK-NEXT:  [[AO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  [[CO:%.*]] = load i32, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK-NEXT:  [[BS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 4) to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[AS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[CS:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  [[C:%.*]] = bitcast void (i32, i32)* %0 to i8*
; CHECK-NEXT:  call void @__dfso_custom_cb_without_ret(void (void (i32, i32)*, i32, i32, i[[#SBITS]], i[[#SBITS]], i32, i32)* @"dfst0$custom_cb_without_ret", i8* [[C]], i32 %1, i32 %2, i[[#SBITS]] zeroext [[CS]], i[[#SBITS]] zeroext [[AS]], i[[#SBITS]] zeroext [[BS]], i32 zeroext [[CO]], i32 zeroext [[AO]], i32 zeroext [[BO]])
; CHECK-NEXT:  ret void

; CHECK: declare void @__dfso_custom_without_ret(i32, i32, i[[#SBITS]], i[[#SBITS]], i32, i32)

; CHECK: declare i32 @__dfso_custom_with_ret(i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i32, i32, i32*)

; CHECK: declare i32 @__dfso_custom_cb_with_ret(i32 (i32 (i32, i32)*, i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i32, i32, i32*)*, i8*, i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i32, i32, i32, i32*)

; CHECK: define linkonce_odr i32 @"dfst0$custom_cb_with_ret"(i32 (i32, i32)* %0, i32 %1, i32 %2, i[[#SBITS]] %3, i[[#SBITS]] %4, i[[#SBITS]]* %5, i32 %6, i32 %7, i32* %8)
; CHECK:   store i32 %6, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK-NEXT:  store i[[#SBITS]] %3, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  store i32 %7, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  store i[[#SBITS]] %4, i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
; CHECK-NEXT:  %9 = call i32 %0(i32 %1, i32 %2)
; CHECK-NEXT:  %_dfsret = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  %_dfsret_o = load i32, i32* @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT:  store i[[#SBITS]] %_dfsret, i[[#SBITS]]* %5, align [[#SBYTES]]
; CHECK-NEXT:  store i32 %_dfsret_o, i32* %8, align 4
; CHECK-NEXT:  ret i32 %9

; CHECK: declare void @__dfso_custom_cb_without_ret(void (void (i32, i32)*, i32, i32, i[[#SBITS]], i[[#SBITS]], i32, i32)*, i8*, i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]], i32, i32, i32)

; CHECK: define linkonce_odr void @"dfst0$custom_cb_without_ret"(void (i32, i32)* %0, i32 %1, i32 %2, i[[#SBITS]] %3, i[[#SBITS]] %4, i32 %5, i32 %6)
; CHECK:  store i32 %5, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 0), align 4
; CHECK-NEXT:  store i[[#SBITS]] %3, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align 2
; CHECK-NEXT:  store i32 %6, i32* getelementptr inbounds ([200 x i32], [200 x i32]* @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  store i[[#SBITS]] %4, i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align 2
; CHECK-NEXT:  call void %0(i32 %1, i32 %2)
; CHECK-NEXT:  ret void

; CHECK: declare void @__dfso_custom_varg_without_ret(i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i32, i32, i32*, ...)

; CHECK: declare i32 @__dfso_custom_varg_with_ret(i32, i32, i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*, i[[#SBITS]]*, i32, i32, i32*, i32*, ...)