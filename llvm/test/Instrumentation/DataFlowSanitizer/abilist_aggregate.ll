; RUN: opt < %s -passes=dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: define { i1, i7 } @functional({ i32, i1 } %a, [2 x i7] %b)
define {i1, i7} @functional({i32, i1} %a, [2 x i7] %b) {
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7} @call_functional({i32, i1} %a, [2 x i7] %b) {
  ; CHECK-LABEL: @call_functional.dfsan
  ; CHECK-NEXT: %[[#REG:]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN:2]]
  ; CHECK-NEXT: %[[#REG+1]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK-NEXT: %[[#REG+2]] = extractvalue { i8, i8 } %[[#REG+1]], 0
  ; CHECK-NEXT: %[[#REG+3]] = extractvalue { i8, i8 } %[[#REG+1]], 1
  ; CHECK-NEXT: %[[#REG+4]] = or i8 %[[#REG+2]], %[[#REG+3]]
  ; CHECK-NEXT: %[[#REG+5]] = extractvalue [2 x i8] %[[#REG]], 0
  ; CHECK-NEXT: %[[#REG+6]] = extractvalue [2 x i8] %[[#REG]], 1
  ; CHECK-NEXT: %[[#REG+7]] = or i8 %[[#REG+5]], %[[#REG+6]]
  ; CHECK-NEXT: %[[#REG+8]] = or i8 %[[#REG+4]], %[[#REG+7]]
  ; CHECK-NEXT: %[[#REG+9]] = insertvalue { i8, i8 } undef, i8 %[[#REG+8]], 0
  ; CHECK-NEXT: %[[#REG+10]] = insertvalue { i8, i8 } %[[#REG+9]], i8 %[[#REG+8]], 1
  ; CHECK: store { i8, i8 } %[[#REG+10]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  %r = call {i1, i7} @functional({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

; CHECK: define { i1, i7 } @discard({ i32, i1 } %a, [2 x i7] %b)
define {i1, i7} @discard({i32, i1} %a, [2 x i7] %b) {
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7} @call_discard({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: @call_discard.dfsan
  ; CHECK: store { i8, i8 } zeroinitializer, ptr @__dfsan_retval_tls, align 2

  %r = call {i1, i7} @discard({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

; CHECK: define { i1, i7 } @uninstrumented({ i32, i1 } %a, [2 x i7] %b)
define {i1, i7} @uninstrumented({i32, i1} %a, [2 x i7] %b) {
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7} @call_uninstrumented({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: @call_uninstrumented.dfsan
  ; CHECK: call void @__dfsan_unimplemented
  ; CHECK: store { i8, i8 } zeroinitializer, ptr @__dfsan_retval_tls, align 2

  %r = call {i1, i7} @uninstrumented({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @call_custom_with_ret({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: @call_custom_with_ret.dfsan
  ; CHECK: %labelreturn = alloca i8, align 1
  ; CHECK: [[B:%.*]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i8, i8 } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i8, i8 } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i8 [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i8] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i8] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i8 [[B0]], [[B1]]
  ; CHECK: [[R:%.*]] = call { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 } %a, [2 x i7] %b, i8 zeroext [[A01]], i8 zeroext [[B01]], ptr %labelreturn)
  ; CHECK: [[RE:%.*]] = load i8, ptr %labelreturn, align 1
  ; CHECK: [[RS0:%.*]] = insertvalue { i8, i8 } undef, i8 [[RE]], 0
  ; CHECK: [[RS1:%.*]] = insertvalue { i8, i8 } [[RS0]], i8 [[RE]], 1
  ; CHECK: store { i8, i8 } [[RS1]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK: ret { i1, i7 } [[R]]

  %r = call {i1, i7} @custom_with_ret({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define void @call_custom_without_ret({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: @call_custom_without_ret.dfsan
  ; CHECK: [[B:%.*]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i8, i8 } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i8, i8 } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i8 [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i8] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i8] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i8 [[B0]], [[B1]]
  ; CHECK: call void @__dfsw_custom_without_ret({ i32, i1 } %a, [2 x i7] %b, i8 zeroext [[A01]], i8 zeroext [[B01]])

  call void @custom_without_ret({i32, i1} %a, [2 x i7] %b)
  ret void
}

define void @call_custom_varg({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: @call_custom_varg.dfsan
  ; CHECK: [[B:%.*]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN:2]]
  ; CHECK: %labelva = alloca [1 x i8], align 1
  ; CHECK: [[A:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i8, i8 } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i8, i8 } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i8 [[A0]], [[A1]]
  ; CHECK: [[V0:%.*]] = getelementptr inbounds nuw [1 x i8], ptr %labelva, i32 0, i32 0
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i8] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i8] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i8 [[B0]], [[B1]]
  ; CHECK: store i8 [[B01]], ptr [[V0]], align 1
  ; CHECK: [[V:%.*]] = getelementptr inbounds nuw [1 x i8], ptr %labelva, i32 0, i32 0
  ; CHECK: call void ({ i32, i1 }, i8, ptr, ...) @__dfsw_custom_varg({ i32, i1 } %a, i8 zeroext [[A01]], ptr [[V]], [2 x i7] %b)

  call void ({i32, i1}, ...) @custom_varg({i32, i1} %a, [2 x i7] %b)
  ret void
}

define {i1, i7} @call_custom_cb({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define { i1, i7 } @call_custom_cb.dfsan({ i32, i1 } %a, [2 x i7] %b) {
  ; CHECK: %labelreturn = alloca i8, align 1
  ; CHECK: [[B:%.*]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i8, i8 } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i8, i8 } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i8 [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i8] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i8] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i8 [[B0]], [[B1]]
  ; CHECK: [[R:%.*]]  = call { i1, i7 } @__dfsw_custom_cb(ptr @cb.dfsan, { i32, i1 } %a, [2 x i7] %b, i8 zeroext 0, i8 zeroext [[A01]], i8 zeroext [[B01]], ptr %labelreturn)
  ; CHECK: [[RE:%.*]] = load i8, ptr %labelreturn, align 1
  ; CHECK: [[RS0:%.*]] = insertvalue { i8, i8 } undef, i8 [[RE]], 0
  ; CHECK: [[RS1:%.*]] = insertvalue { i8, i8 } [[RS0]], i8 [[RE]], 1
  ; CHECK: store { i8, i8 } [[RS1]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  %r = call {i1, i7} @custom_cb(ptr @cb, {i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @custom_cb(ptr %cb, {i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define { i1, i7 } @custom_cb(ptr %cb, { i32, i1 } %a, [2 x i7] %b)

  %r = call {i1, i7} %cb({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @cb({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define { i1, i7 } @cb.dfsan({ i32, i1 } %a, [2 x i7] %b)
  ; CHECK: [[BL:%.*]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN:2]]
  ; CHECK: [[AL:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK: [[AL1:%.*]] = extractvalue { i8, i8 } [[AL]], 1
  ; CHECK: [[BL0:%.*]] = extractvalue [2 x i8] [[BL]], 0
  ; CHECK: [[RL0:%.*]] = insertvalue { i8, i8 } zeroinitializer, i8 [[AL1]], 0
  ; CHECK: [[RL:%.*]] = insertvalue { i8, i8 } [[RL0]], i8 [[BL0]], 1
  ; CHECK: store { i8, i8 } [[RL]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define ptr @ret_custom() {
  ; CHECK: @ret_custom.dfsan
  ; CHECK: store i8 0, ptr @__dfsan_retval_tls, align 2
  ; CHECK: ret {{.*}} @"dfsw$custom_with_ret"
  ret ptr @custom_with_ret
}

; CHECK: define linkonce_odr { i1, i7 } @"dfsw$custom_cb"(ptr %0, { i32, i1 } %1, [2 x i7] %2) {
; CHECK: %labelreturn = alloca i8, align 1
; COMM: TODO simplify the expression [[#mul(2,SBYTES) + max(SBYTES,2)]] to
; COMM: [[#mul(3,SBYTES)]], if shadow-tls-alignment is updated to match shadow
; COMM: width bytes.
; CHECK: [[B:%.*]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 4), align [[ALIGN:2]]
; CHECK: [[A:%.*]] = load { i8, i8 }, ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN]]
; CHECK: [[CB:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN]]
; CHECK: [[A0:%.*]] = extractvalue { i8, i8 } [[A]], 0
; CHECK: [[A1:%.*]] = extractvalue { i8, i8 } [[A]], 1
; CHECK: [[A01:%.*]] = or i8 [[A0]], [[A1]]
; CHECK: [[B0:%.*]] = extractvalue [2 x i8] [[B]], 0
; CHECK: [[B1:%.*]] = extractvalue [2 x i8] [[B]], 1
; CHECK: [[B01:%.*]] = or i8 [[B0]], [[B1]]
; CHECK: [[R:%.*]]  = call { i1, i7 } @__dfsw_custom_cb(ptr %0, { i32, i1 } %1, [2 x i7] %2, i8 zeroext [[CB]], i8 zeroext [[A01]], i8 zeroext [[B01]], ptr %labelreturn)
; CHECK: [[RE:%.*]] = load i8, ptr %labelreturn, align 1
; CHECK: [[RS0:%.*]] = insertvalue { i8, i8 } undef, i8 [[RE]], 0
; CHECK: [[RS1:%.*]] = insertvalue { i8, i8 } [[RS0]], i8 [[RE]], 1
; CHECK: store { i8, i8 } [[RS1]], ptr @__dfsan_retval_tls, align [[ALIGN]]

define {i1, i7} @custom_with_ret({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define linkonce_odr { i1, i7 } @"dfsw$custom_with_ret"({ i32, i1 } %0, [2 x i7] %1)
  ; CHECK: %labelreturn = alloca i8, align 1
  ; CHECK: [[B:%.*]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i8, i8 } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i8, i8 } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i8 [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i8] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i8] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i8 [[B0]], [[B1]]
  ; CHECK: [[R:%.*]] = call { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 } %0, [2 x i7] %1, i8 zeroext [[A01]], i8 zeroext [[B01]], ptr %labelreturn)
  ; CHECK: [[RE:%.*]] = load i8, ptr %labelreturn, align 1
  ; CHECK: [[RS0:%.*]] = insertvalue { i8, i8 } undef, i8 [[RE]], 0
  ; CHECK: [[RS1:%.*]] = insertvalue { i8, i8 } [[RS0]], i8 [[RE]], 1
  ; CHECK: store { i8, i8 } [[RS1]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; CHECK: ret { i1, i7 } [[R]]
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define void @custom_without_ret({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define linkonce_odr void @"dfsw$custom_without_ret"({ i32, i1 } %0, [2 x i7] %1)
  ; CHECK: [[B:%.*]] = load [2 x i8], ptr getelementptr (i8, ptr @__dfsan_arg_tls, i64 2), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i8, i8 } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i8, i8 } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i8 [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i8] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i8] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i8 [[B0]], [[B1]]
  ; CHECK: call void @__dfsw_custom_without_ret({ i32, i1 } %0, [2 x i7] %1, i8 zeroext [[A01]], i8 zeroext [[B01]])
  ; CHECK: ret
  ret void
}

define void @custom_varg({i32, i1} %a, ...) {
  ; CHECK: define linkonce_odr void @"dfsw$custom_varg"({ i32, i1 } %0, ...)
  ; CHECK: call void @__dfsan_vararg_wrapper
  ; CHECK: unreachable
  ret void
}

; CHECK: declare { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 }, [2 x i7], i8, i8, ptr)
; CHECK: declare void @__dfsw_custom_without_ret({ i32, i1 }, [2 x i7], i8, i8)
; CHECK: declare void @__dfsw_custom_varg({ i32, i1 }, i8, ptr, ...)

; CHECK: declare { i1, i7 } @__dfsw_custom_cb(ptr, { i32, i1 }, [2 x i7], i8, i8, i8, ptr)
