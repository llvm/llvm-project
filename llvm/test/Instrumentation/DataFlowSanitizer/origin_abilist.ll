; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1  -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
define i32 @discard(i32 %a, i32 %b) {
  ret i32 0
}

define i32 @call_discard(i32 %a, i32 %b) {
  ; CHECK: @call_discard.dfsan
  ; CHECK: %r = call i32 @discard(i32 %a, i32 %b)
  ; CHECK: store i32 0, ptr @__dfsan_retval_origin_tls, align 4
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
  ; CHECK-LABEL: @call_functional.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls
  ; CHECK: [[RO:%.*]] = select i1 {{.*}}, i32 [[BO]], i32 [[AO]]
  ; CHECK: store i32 [[RO]], ptr @__dfsan_retval_origin_tls, align 4

  %r = call i32 @functional(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @uninstrumented(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

define i32 @call_uninstrumented(i32 %a, i32 %b) {
  ; CHECK-LABEL: @call_uninstrumented.dfsan
  ; CHECK: %r = call i32 @uninstrumented(i32 %a, i32 %b)
  ; CHECK: store i32 0, ptr @__dfsan_retval_origin_tls, align 4
  ; CHECK: ret i32 %r

  %r = call i32 @uninstrumented(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @g(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

@discardg = alias i32 (i32, i32), ptr @g

define i32 @call_discardg(i32 %a, i32 %b) {
  ; CHECK: @call_discardg.dfsan
  ; CHECK: %r = call i32 @discardg(i32 %a, i32 %b)
  ; CHECK: store i32 0, ptr @__dfsan_retval_origin_tls, align 4
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

define i32 @custom_cb_with_ret(ptr %cb, i32 %a, i32 %b) {
  %r = call i32 %cb(i32 %a, i32 %b)
  ret i32 %r
}

define i32 @cb_with_ret(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

define void @custom_cb_without_ret(ptr %cb, i32 %a, i32 %b) {
  call void %cb(i32 %a, i32 %b)
  ret void
}

define void @cb_without_ret(i32 %a, i32 %b) {
  ret void
}

define ptr @ret_custom() {
  ; CHECK: @ret_custom.dfsan
  ; CHECK: store i32 0, ptr @__dfsan_retval_origin_tls, align 4

  ret ptr @custom_with_ret
}

define void @call_custom_without_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_without_ret.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK: call void @__dfso_custom_without_ret(i32 %a, i32 %b, i8 zeroext [[AS]], i8 zeroext [[BS]], i32 zeroext [[AO]], i32 zeroext [[BO]])
  ; CHECK-NEXT: ret void

  call void @custom_without_ret(i32 %a, i32 %b)
  ret void
}

define i32 @call_custom_with_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_with_ret.dfsan
  ; CHECK: %originreturn = alloca i32, align 4
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: %labelreturn = alloca i8, align 1
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK: {{.*}} = call i32 @__dfso_custom_with_ret(i32 %a, i32 %b, i8 zeroext [[AS]], i8 zeroext [[BS]], ptr %labelreturn, i32 zeroext [[AO]], i32 zeroext [[BO]], ptr %originreturn)
  ; CHECK: [[RS:%.*]] = load i8, ptr %labelreturn, align 1
  ; CHECK: [[RO:%.*]] = load i32, ptr %originreturn, align 4
  ; CHECK: store i8 [[RS]], ptr @__dfsan_retval_tls, align 2
  ; CHECK: store i32 [[RO]], ptr @__dfsan_retval_origin_tls, align 4

  %r = call i32 @custom_with_ret(i32 %a, i32 %b)
  ret i32 %r
}

define void @call_custom_varg_without_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_varg_without_ret.dfsan
  ; CHECK: %originva = alloca [1 x i32], align 4
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: %labelva = alloca [1 x i8], align 1
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK: [[VS0:%.*]] = getelementptr inbounds [1 x i8], ptr %labelva, i32 0, i32 0
  ; CHECK: store i8 [[AS]], ptr [[VS0]], align 1
  ; CHECK: [[VS0:%.*]] = getelementptr inbounds [1 x i8], ptr %labelva, i32 0, i32 0
  ; CHECK: [[VO0:%.*]] = getelementptr inbounds [1 x i32], ptr %originva, i32 0, i32 0
  ; CHECK: store i32 [[AO]], ptr [[VO0]], align 4
  ; CHECK: [[VO0:%.*]] = getelementptr inbounds [1 x i32], ptr %originva, i32 0, i32 0
  ; CHECK: call void (i32, i32, i8, i8, ptr, i32, i32, ptr, ...) @__dfso_custom_varg_without_ret(i32 %a, i32 %b, i8 zeroext [[AS]], i8 zeroext [[BS]], ptr [[VS0]], i32 zeroext [[AO]], i32 zeroext [[BO]], ptr [[VO0]], i32 %a)
  ; CHECK-NEXT: ret void

  call void (i32, i32, ...) @custom_varg_without_ret(i32 %a, i32 %b, i32 %a)
  ret void
}

define i32 @call_custom_varg_with_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_varg_with_ret.dfsan
  ; CHECK: %originreturn = alloca i32, align 4
  ; CHECK: %originva = alloca [1 x i32], align 4
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls
  ; CHECK: %labelreturn = alloca i8, align 1
  ; CHECK: %labelva = alloca [1 x i8], align 1
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK: [[VS0:%.*]] = getelementptr inbounds [1 x i8], ptr %labelva, i32 0, i32 0
  ; CHECK: store i8 [[BS]], ptr [[VS0]], align 1
  ; CHECK: [[VS0:%.*]] = getelementptr inbounds [1 x i8], ptr %labelva, i32 0, i32 0
  ; CHECK: [[VO0:%.*]] = getelementptr inbounds [1 x i32], ptr %originva, i32 0, i32 0
  ; CHECK: store i32 [[BO]], ptr [[VO0]], align 4
  ; CHECK: [[VO0:%.*]] = getelementptr inbounds [1 x i32], ptr %originva, i32 0, i32 0
  ; CHECK: {{.*}} = call i32 (i32, i32, i8, i8, ptr, ptr, i32, i32, ptr, ptr, ...) @__dfso_custom_varg_with_ret(i32 %a, i32 %b, i8 zeroext [[AS]], i8 zeroext [[BS]], ptr [[VS0]], ptr %labelreturn, i32 zeroext [[AO]], i32 zeroext [[BO]], ptr [[VO0]], ptr %originreturn, i32 %b)
  ; CHECK: [[RS:%.*]] = load i8, ptr %labelreturn, align 1
  ; CHECK: [[RO:%.*]] = load i32, ptr %originreturn, align 4
  ; CHECK: store i8 [[RS]], ptr @__dfsan_retval_tls, align 2
  ; CHECK: store i32 [[RO]], ptr @__dfsan_retval_origin_tls, align 4

  %r = call i32 (i32, i32, ...) @custom_varg_with_ret(i32 %a, i32 %b, i32 %b)
  ret i32 %r
}

define i32 @call_custom_cb_with_ret(i32 %a, i32 %b) {
  ; CHECK: @call_custom_cb_with_ret.dfsan
  ; CHECK: %originreturn = alloca i32, align 4
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: %labelreturn = alloca i8, align 1
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK: {{.*}} = call i32 @__dfso_custom_cb_with_ret(ptr @cb_with_ret.dfsan, i32 %a, i32 %b, i8 zeroext 0, i8 zeroext [[AS]], i8 zeroext [[BS]], ptr %labelreturn, i32 zeroext 0, i32 zeroext [[AO]], i32 zeroext [[BO]], ptr %originreturn)
  ; CHECK: [[RS:%.*]] = load i8, ptr %labelreturn, align 1
  ; CHECK: [[RO:%.*]] = load i32, ptr %originreturn, align 4
  ; CHECK: store i8 [[RS]], ptr @__dfsan_retval_tls, align 2
  ; CHECK: store i32 [[RO]], ptr @__dfsan_retval_origin_tls, align 4

  %r = call i32 @custom_cb_with_ret(ptr @cb_with_ret, i32 %a, i32 %b)
  ret i32 %r
}

define void @call_custom_cb_without_ret(i32 %a, i32 %b) {
  ; CHECK-LABEL: @call_custom_cb_without_ret.dfsan
  ; CHECK: [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
  ; CHECK: [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
  ; CHECK: [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
  ; CHECK: [[AS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
  ; CHECK: call void @__dfso_custom_cb_without_ret(ptr @cb_without_ret.dfsan, i32 %a, i32 %b, i8 zeroext 0, i8 zeroext [[AS]], i8 zeroext [[BS]], i32 zeroext 0, i32 zeroext [[AO]], i32 zeroext [[BO]])
  ; CHECK-NEXT: ret void

  call void @custom_cb_without_ret(ptr @cb_without_ret, i32 %a, i32 %b)
  ret void
}

; CHECK: define i32 @discardg(i32 %0, i32 %1)
; CHECK: [[R:%.*]] = call i32 @g.dfsan
; CHECK-NEXT: %_dfsret = load i8, ptr @__dfsan_retval_tls, align 2
; CHECK-NEXT: %_dfsret_o = load i32, ptr @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT: ret i32 [[R]]

; CHECK: define linkonce_odr void @"dfso$custom_without_ret"(i32 %0, i32 %1)
; CHECK:  [[BO:%.*]]  = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  [[AO:%.*]]  = load i32, ptr @__dfsan_arg_origin_tls, align 4
; CHECK-NEXT:  [[BS:%.*]]  = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
; CHECK-NEXT:  [[AS:%.*]]  = load i8, ptr @__dfsan_arg_tls, align 2
; CHECK-NEXT:  call void @__dfso_custom_without_ret(i32 %0, i32 %1, i8 zeroext [[AS]], i8 zeroext [[BS]], i32 zeroext [[AO]], i32 zeroext [[BO]])
; CHECK-NEXT:  ret void

; CHECK: define linkonce_odr i32 @"dfso$custom_with_ret"(i32 %0, i32 %1)
; CHECK:  %originreturn = alloca i32, align 4
; CHECK-NEXT:  [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  [[AO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
; CHECK-NEXT:  %labelreturn = alloca i8, align 1
; CHECK-NEXT:  [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
; CHECK-NEXT:  [[AS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
; CHECK-NEXT:  [[R:%.*]] = call i32 @__dfso_custom_with_ret(i32 %0, i32 %1, i8 zeroext [[AS]], i8 zeroext [[BS]], ptr %labelreturn, i32 zeroext [[AO]], i32 zeroext [[BO]], ptr %originreturn)
; CHECK-NEXT:  [[RS:%.*]] = load i8, ptr %labelreturn, align 1
; CHECK-NEXT:  [[RO:%.*]] = load i32, ptr %originreturn, align 4
; CHECK-NEXT:  store i8 [[RS]], ptr @__dfsan_retval_tls, align 2
; CHECK-NEXT:  store i32 [[RO]], ptr @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT:  ret i32 [[R]]

; CHECK: define linkonce_odr void @"dfso$custom_varg_without_ret"(i32 %0, i32 %1, ...)
; CHECK:  call void @__dfsan_vararg_wrapper(ptr @0)
; CHECK-NEXT:  unreachable

; CHECK: define linkonce_odr i32 @"dfso$custom_varg_with_ret"(i32 %0, i32 %1, ...)
; CHECK:  call void @__dfsan_vararg_wrapper(ptr @1)
; CHECK-NEXT:  unreachable

; CHECK: define linkonce_odr i32 @"dfso$custom_cb_with_ret"(ptr %0, i32 %1, i32 %2)
; CHECK:  %originreturn = alloca i32, align 4
; CHECK-NEXT:  [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 2), align 4
; CHECK-NEXT:  [[AO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  [[CO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
; CHECK-NEXT:  %labelreturn = alloca i8, align 1
; CHECK-NEXT:  [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align 2
; CHECK-NEXT:  [[AS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
; CHECK-NEXT:  [[CS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
; CHECK-NEXT:  [[R:%.*]] = call i32 @__dfso_custom_cb_with_ret(ptr %0, i32 %1, i32 %2, i8 zeroext [[CS]], i8 zeroext [[AS]], i8 zeroext [[BS]], ptr %labelreturn, i32 zeroext [[CO]], i32 zeroext [[AO]], i32 zeroext [[BO]], ptr %originreturn)
; CHECK-NEXT:  [[RS:%.*]] = load i8, ptr %labelreturn, align 1
; CHECK-NEXT:  [[RO:%.*]] = load i32, ptr %originreturn, align 4
; CHECK-NEXT:  store i8 [[RS]], ptr @__dfsan_retval_tls, align 2
; CHECK-NEXT:  store i32 [[RO]], ptr @__dfsan_retval_origin_tls, align 4
; CHECK-NEXT:  ret i32 [[R]]

; CHECK: define linkonce_odr void @"dfso$custom_cb_without_ret"(ptr %0, i32 %1, i32 %2)
; CHECK:   [[BO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 2), align 4
; CHECK-NEXT:  [[AO:%.*]] = load i32, ptr getelementptr inbounds ([200 x i32], ptr @__dfsan_arg_origin_tls, i64 0, i64 1), align 4
; CHECK-NEXT:  [[CO:%.*]] = load i32, ptr @__dfsan_arg_origin_tls, align 4
; CHECK-NEXT:  [[BS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align 2
; CHECK-NEXT:  [[AS:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align 2
; CHECK-NEXT:  [[CS:%.*]] = load i8, ptr @__dfsan_arg_tls, align 2
; CHECK-NEXT:  call void @__dfso_custom_cb_without_ret(ptr %0, i32 %1, i32 %2, i8 zeroext [[CS]], i8 zeroext [[AS]], i8 zeroext [[BS]], i32 zeroext [[CO]], i32 zeroext [[AO]], i32 zeroext [[BO]])
; CHECK-NEXT:  ret void

; CHECK: declare void @__dfso_custom_without_ret(i32, i32, i8, i8, i32, i32)

; CHECK: declare i32 @__dfso_custom_with_ret(i32, i32, i8, i8, ptr, i32, i32, ptr)

; CHECK: declare i32 @__dfso_custom_cb_with_ret(ptr, i32, i32, i8, i8, i8, ptr, i32, i32, i32, ptr)

; CHECK: declare void @__dfso_custom_cb_without_ret(ptr, i32, i32, i8, i8, i8, i32, i32, i32)

; CHECK: declare void @__dfso_custom_varg_without_ret(i32, i32, i8, i8, ptr, i32, i32, ptr, ...)

; CHECK: declare i32 @__dfso_custom_varg_with_ret(i32, i32, i8, i8, ptr, ptr, i32, i32, ptr, ptr, ...)
