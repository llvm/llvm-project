; RUN: opt < %s -dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

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
  ; CHECK-NEXT: %[[#REG:]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; CHECK-NEXT: %[[#REG+1]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK-NEXT: %[[#REG+2]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } %[[#REG+1]], 0
  ; CHECK-NEXT: %[[#REG+3]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } %[[#REG+1]], 1
  ; CHECK-NEXT: %[[#REG+4]] = or i[[#SBITS]] %[[#REG+2]], %[[#REG+3]]
  ; CHECK-NEXT: %[[#REG+5]] = extractvalue [2 x i[[#SBITS]]] %[[#REG]], 0
  ; CHECK-NEXT: %[[#REG+6]] = extractvalue [2 x i[[#SBITS]]] %[[#REG]], 1
  ; CHECK-NEXT: %[[#REG+7]] = or i[[#SBITS]] %[[#REG+5]], %[[#REG+6]]
  ; CHECK-NEXT: %[[#REG+8]] = or i[[#SBITS]] %[[#REG+4]], %[[#REG+7]]
  ; CHECK-NEXT: %[[#REG+9]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] %[[#REG+8]], 0
  ; CHECK-NEXT: %[[#REG+10]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } %[[#REG+9]], i[[#SBITS]] %[[#REG+8]], 1
  ; CHECK: store { i[[#SBITS]], i[[#SBITS]] } %[[#REG+10]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]

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
  ; CHECK: store { i[[#SBITS]], i[[#SBITS]] } zeroinitializer, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align 2

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
  ; CHECK: store { i[[#SBITS]], i[[#SBITS]] } zeroinitializer, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align 2

  %r = call {i1, i7} @uninstrumented({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @call_custom_with_ret({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: @call_custom_with_ret.dfsan
  ; CHECK: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; CHECK: [[R:%.*]] = call { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 } %a, [2 x i7] %b, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]], i[[#SBITS]]* %labelreturn)
  ; CHECK: [[RE:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; CHECK: [[RS0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] [[RE]], 0
  ; CHECK: [[RS1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RS0]], i[[#SBITS]] [[RE]], 1
  ; CHECK: store { i[[#SBITS]], i[[#SBITS]] } [[RS1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: ret { i1, i7 } [[R]]

  %r = call {i1, i7} @custom_with_ret({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define void @call_custom_without_ret({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: @call_custom_without_ret.dfsan
  ; CHECK: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; CHECK: call void @__dfsw_custom_without_ret({ i32, i1 } %a, [2 x i7] %b, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]])

  call void @custom_without_ret({i32, i1} %a, [2 x i7] %b)
  ret void
}

define void @call_custom_varg({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: @call_custom_varg.dfsan
  ; CHECK: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; CHECK: %labelva = alloca [1 x i[[#SBITS]]], align [[#SBYTES]]
  ; CHECK: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; CHECK: [[V0:%.*]] = getelementptr inbounds [1 x i[[#SBITS]]], [1 x i[[#SBITS]]]* %labelva, i32 0, i32 0
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; CHECK: store i[[#SBITS]] [[B01]], i[[#SBITS]]* [[V0]], align [[#SBYTES]]
  ; CHECK: [[V:%.*]] = getelementptr inbounds [1 x i[[#SBITS]]], [1 x i[[#SBITS]]]* %labelva, i32 0, i32 0
  ; CHECK: call void ({ i32, i1 }, i[[#SBITS]], i[[#SBITS]]*, ...) @__dfsw_custom_varg({ i32, i1 } %a, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]]* [[V]], [2 x i7] %b)

  call void ({i32, i1}, ...) @custom_varg({i32, i1} %a, [2 x i7] %b)
  ret void
}

define {i1, i7} @call_custom_cb({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define { i1, i7 } @call_custom_cb.dfsan({ i32, i1 } %a, [2 x i7] %b) {
  ; CHECK: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; CHECK: [[R:%.*]]  = call { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)* @"dfst0$custom_cb", i8* bitcast ({ i1, i7 } ({ i32, i1 }, [2 x i7])* @cb.dfsan to i8*), { i32, i1 } %a, [2 x i7] %b, i[[#SBITS]] zeroext 0, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]], i[[#SBITS]]* %labelreturn)
  ; CHECK: [[RE:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; CHECK: [[RS0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] [[RE]], 0
  ; CHECK: [[RS1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RS0]], i[[#SBITS]] [[RE]], 1
  ; CHECK: store { i[[#SBITS]], i[[#SBITS]] } [[RS1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]

  %r = call {i1, i7} @custom_cb({i1, i7} ({i32, i1}, [2 x i7])* @cb, {i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @custom_cb({i1, i7} ({i32, i1}, [2 x i7])* %cb, {i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define { i1, i7 } @custom_cb({ i1, i7 } ({ i32, i1 }, [2 x i7])* %cb, { i32, i1 } %a, [2 x i7] %b)

  %r = call {i1, i7} %cb({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @cb({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define { i1, i7 } @cb.dfsan({ i32, i1 } %a, [2 x i7] %b)
  ; CHECK: [[BL:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; CHECK: [[AL:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: [[AL1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[AL]], 1
  ; CHECK: [[BL0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[BL]], 0
  ; CHECK: [[RL0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } zeroinitializer, i[[#SBITS]] [[AL1]], 0
  ; CHECK: [[RL:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RL0]], i[[#SBITS]] [[BL0]], 1
  ; CHECK: store { i[[#SBITS]], i[[#SBITS]] } [[RL]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]

  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7}  ({i32, i1}, [2 x i7])* @ret_custom() {
  ; CHECK: @ret_custom.dfsan
  ; CHECK: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; CHECK: ret {{.*}} @"dfsw$custom_with_ret"
  ret {i1, i7}  ({i32, i1}, [2 x i7])* @custom_with_ret
}

; CHECK: define linkonce_odr { i1, i7 } @"dfsw$custom_cb"({ i1, i7 } ({ i32, i1 }, [2 x i7])* %0, { i32, i1 } %1, [2 x i7] %2) {
; CHECK: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
; COMM: TODO simplify the expression [[#mul(2,SBYTES) + max(SBYTES,2)]] to
; COMM: [[#mul(3,SBYTES)]], if shadow-tls-alignment is updated to match shadow
; COMM: width bytes.
; CHECK: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES) + max(SBYTES,2)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
; CHECK: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
; CHECK: [[CB:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
; CHECK: [[CAST:%.*]] = bitcast { i1, i7 } ({ i32, i1 }, [2 x i7])* %0 to i8*
; CHECK: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
; CHECK: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
; CHECK: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
; CHECK: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
; CHECK: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
; CHECK: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
; CHECK: [[R:%.*]]  = call { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)* @"dfst0$custom_cb", i8* [[CAST]], { i32, i1 } %1, [2 x i7] %2, i[[#SBITS]] zeroext [[CB]], i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]], i[[#SBITS]]* %labelreturn)
; CHECK: [[RE:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
; CHECK: [[RS0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] [[RE]], 0
; CHECK: [[RS1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RS0]], i[[#SBITS]] [[RE]], 1
; CHECK: store { i[[#SBITS]], i[[#SBITS]] } [[RS1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]


define {i1, i7} @custom_with_ret({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define linkonce_odr { i1, i7 } @"dfsw$custom_with_ret"({ i32, i1 } %0, [2 x i7] %1)
  ; CHECK: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; CHECK: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; CHECK: [[R:%.*]] = call { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 } %0, [2 x i7] %1, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]], i[[#SBITS]]* %labelreturn)
  ; CHECK: [[RE:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; CHECK: [[RS0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] [[RE]], 0
  ; CHECK: [[RS1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RS0]], i[[#SBITS]] [[RE]], 1
  ; CHECK: store { i[[#SBITS]], i[[#SBITS]] } [[RS1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: ret { i1, i7 } [[R]]
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define void @custom_without_ret({i32, i1} %a, [2 x i7] %b) {
  ; CHECK: define linkonce_odr void @"dfsw$custom_without_ret"({ i32, i1 } %0, [2 x i7] %1)
  ; CHECK: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; CHECK: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; CHECK: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; CHECK: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; CHECK: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; CHECK: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; CHECK: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; CHECK: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; CHECK: call void @__dfsw_custom_without_ret({ i32, i1 } %0, [2 x i7] %1, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]])
  ; CHECK: ret
  ret void
}

define void @custom_varg({i32, i1} %a, ...) {
  ; CHECK: define linkonce_odr void @"dfsw$custom_varg"({ i32, i1 } %0, ...)
  ; CHECK: call void @__dfsan_vararg_wrapper
  ; CHECK: unreachable
  ret void
}

; CHECK: declare { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)
; CHECK: declare void @__dfsw_custom_without_ret({ i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]])
; CHECK: declare void @__dfsw_custom_varg({ i32, i1 }, i[[#SBITS]], i[[#SBITS]]*, ...)

; CHECK: declare { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)*, i8*, { i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)

; CHECK: define linkonce_odr { i1, i7 } @"dfst0$custom_cb"({ i1, i7 } ({ i32, i1 }, [2 x i7])* %0, { i32, i1 } %1, [2 x i7] %2, i[[#SBITS]] %3, i[[#SBITS]] %4, i[[#SBITS]]* %5) {
; CHECK: [[A0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] %3, 0
; CHECK: [[A1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[A0]], i[[#SBITS]] %3, 1
; CHECK: [[B0:%.*]] = insertvalue [2 x i[[#SBITS]]] undef, i[[#SBITS]] %4, 0
; CHECK: [[B1:%.*]] = insertvalue [2 x i[[#SBITS]]] [[B0]], i[[#SBITS]] %4, 1
; CHECK: store { i[[#SBITS]], i[[#SBITS]] } [[A1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN:2]]
; CHECK: store [2 x i[[#SBITS]]] [[B1]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN]]
; CHECK: [[R:%.*]] = call { i1, i7 } %0({ i32, i1 } %1, [2 x i7] %2)
; CHECK: %_dfsret = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
; CHECK: [[RE0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } %_dfsret, 0
; CHECK: [[RE1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } %_dfsret, 1
; CHECK: [[RE01:%.*]] = or i[[#SBITS]] [[RE0]], [[RE1]]
; CHECK: store i[[#SBITS]] [[RE01]], i[[#SBITS]]* %5, align [[#SBYTES]]
; CHECK: ret { i1, i7 } [[R]]
