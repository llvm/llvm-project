; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefixes=CHECK,TLS_ABI
; RUN: opt < %s -dfsan -dfsan-fast-8-labels=true -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefixes=CHECK,TLS_ABI
; RUN: opt < %s -dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefixes=CHECK,LEGACY
; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefixes=CHECK,ARGS_ABI
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

; TLS_ABI: define { i1, i7 } @functional({ i32, i1 } %a, [2 x i7] %b)
; ARGS_ABI: define { i1, i7 } @functional({ i32, i1 } %a, [2 x i7] %b)
define {i1, i7} @functional({i32, i1} %a, [2 x i7] %b) {
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7} @call_functional({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI-LABEL: @"dfs$call_functional"
  ; TLS_ABI-NEXT: %[[#REG:]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; TLS_ABI-NEXT: %[[#REG+1]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI-NEXT: %[[#REG+2]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } %[[#REG+1]], 0
  ; TLS_ABI-NEXT: %[[#REG+3]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } %[[#REG+1]], 1
  ; TLS_ABI-NEXT: %[[#REG+4]] = or i[[#SBITS]] %[[#REG+2]], %[[#REG+3]]
  ; TLS_ABI-NEXT: %[[#REG+5]] = extractvalue [2 x i[[#SBITS]]] %[[#REG]], 0
  ; TLS_ABI-NEXT: %[[#REG+6]] = extractvalue [2 x i[[#SBITS]]] %[[#REG]], 1
  ; TLS_ABI-NEXT: %[[#REG+7]] = or i[[#SBITS]] %[[#REG+5]], %[[#REG+6]]
  ; TLS_ABI-NEXT: %[[#REG+8]] = or i[[#SBITS]] %[[#REG+4]], %[[#REG+7]]
  ; TLS_ABI-NEXT: %[[#REG+9]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] %[[#REG+8]], 0
  ; TLS_ABI-NEXT: %[[#REG+10]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } %[[#REG+9]], i[[#SBITS]] %[[#REG+8]], 1
  ; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } %[[#REG+10]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]

  ; LEGACY: @"dfs$call_functional"
  ; LEGACY: [[B:%.*]] = load i[[#SBITS]], i[[#SBITS]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; LEGACY: [[A:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; LEGACY: [[U:%.*]] = call zeroext i[[#SBITS]] @__dfsan_union(i[[#SBITS]] zeroext [[A]], i[[#SBITS]] zeroext [[B]])
  ; LEGACY: [[PH:%.*]] = phi i[[#SBITS]] [ [[U]], {{.*}} ], [ [[A]], {{.*}} ]
  ; LEGACY: store i[[#SBITS]] [[PH]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]

  ; ARGS_ABI: @"dfs$call_functional"
  ; ARGS_ABI: [[U:%.*]]  = call zeroext i[[#SBITS]] @__dfsan_union(i[[#SBITS]] zeroext %2, i[[#SBITS]] zeroext %3)
  ; ARGS_ABI: [[PH:%.*]] = phi i[[#SBITS]] [ %7, {{.*}} ], [ %2, {{.*}} ]
  ; ARGS_ABI: [[R0:%.*]] = insertvalue { { i1, i7 }, i[[#SBITS]] } undef, { i1, i7 } %r, 0
  ; ARGS_ABI: [[R1:%.*]] = insertvalue { { i1, i7 }, i[[#SBITS]] } [[R0]], i[[#SBITS]] [[PH]], 1
  ; ARGS_ABI: ret { { i1, i7 }, i[[#SBITS]] } [[R1]]

  %r = call {i1, i7} @functional({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

; TLS_ABI: define { i1, i7 } @discard({ i32, i1 } %a, [2 x i7] %b)
define {i1, i7} @discard({i32, i1} %a, [2 x i7] %b) {
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7} @call_discard({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: @"dfs$call_discard"
  ; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } zeroinitializer, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align 2

  ; ARGS_ABI: @"dfs$call_discard"
  ; ARGS_ABI: %r = call { i1, i7 } @discard({ i32, i1 } %0, [2 x i7] %1)
  ; ARGS_ABI: [[R0:%.*]] = insertvalue { { i1, i7 }, i[[#SBITS]] } undef, { i1, i7 } %r, 0
  ; ARGS_ABI: [[R1:%.*]] = insertvalue { { i1, i7 }, i[[#SBITS]] } [[R0]], i[[#SBITS]] 0, 1
  ; ARGS_ABI: ret { { i1, i7 }, i[[#SBITS]] } [[R1]]

  %r = call {i1, i7} @discard({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

; TLS_ABI: define { i1, i7 } @uninstrumented({ i32, i1 } %a, [2 x i7] %b)
define {i1, i7} @uninstrumented({i32, i1} %a, [2 x i7] %b) {
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7} @call_uninstrumented({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: @"dfs$call_uninstrumented"
  ; TLS_ABI: call void @__dfsan_unimplemented
  ; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } zeroinitializer, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align 2

  ; ARGS_ABI: @"dfs$call_uninstrumented"
  ; ARGS_ABI: call void @__dfsan_unimplemented
  ; ARGS_ABI: %r = call { i1, i7 } @uninstrumented({ i32, i1 } %0, [2 x i7] %1)
  ; ARGS_ABI: [[R0:%.*]] = insertvalue { { i1, i7 }, i[[#SBITS]] } undef, { i1, i7 } %r, 0
  ; ARGS_ABI: [[R1:%.*]] = insertvalue { { i1, i7 }, i[[#SBITS]] } [[R0]], i[[#SBITS]] 0, 1
  ; ARGS_ABI: ret { { i1, i7 }, i[[#SBITS]] } [[R1]]

  %r = call {i1, i7} @uninstrumented({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @call_custom_with_ret({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: @"dfs$call_custom_with_ret"
  ; TLS_ABI: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; TLS_ABI: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; TLS_ABI: [[R:%.*]] = call { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 } %a, [2 x i7] %b, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]], i[[#SBITS]]* %labelreturn)
  ; TLS_ABI: [[RE:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; TLS_ABI: [[RS0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] [[RE]], 0
  ; TLS_ABI: [[RS1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RS0]], i[[#SBITS]] [[RE]], 1
  ; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } [[RS1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: ret { i1, i7 } [[R]]

  %r = call {i1, i7} @custom_with_ret({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define void @call_custom_without_ret({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: @"dfs$call_custom_without_ret"
  ; TLS_ABI: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; TLS_ABI: call void @__dfsw_custom_without_ret({ i32, i1 } %a, [2 x i7] %b, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]])

  call void @custom_without_ret({i32, i1} %a, [2 x i7] %b)
  ret void
}

define void @call_custom_varg({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: @"dfs$call_custom_varg"
  ; TLS_ABI: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; TLS_ABI: %labelva = alloca [1 x i[[#SBITS]]], align [[#SBYTES]]
  ; TLS_ABI: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; TLS_ABI: [[V0:%.*]] = getelementptr inbounds [1 x i[[#SBITS]]], [1 x i[[#SBITS]]]* %labelva, i32 0, i32 0
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; TLS_ABI: store i[[#SBITS]] [[B01]], i[[#SBITS]]* [[V0]], align [[#SBYTES]]
  ; TLS_ABI: [[V:%.*]] = getelementptr inbounds [1 x i[[#SBITS]]], [1 x i[[#SBITS]]]* %labelva, i32 0, i32 0
  ; TLS_ABI: call void ({ i32, i1 }, i[[#SBITS]], i[[#SBITS]]*, ...) @__dfsw_custom_varg({ i32, i1 } %a, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]]* [[V]], [2 x i7] %b)

  call void ({i32, i1}, ...) @custom_varg({i32, i1} %a, [2 x i7] %b)
  ret void
}

define {i1, i7} @call_custom_cb({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: define { i1, i7 } @"dfs$call_custom_cb"({ i32, i1 } %a, [2 x i7] %b) {
  ; TLS_ABI: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; TLS_ABI: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; TLS_ABI: [[R:%.*]]  = call { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)* @"dfst0$custom_cb", i8* bitcast ({ i1, i7 } ({ i32, i1 }, [2 x i7])* @"dfs$cb" to i8*), { i32, i1 } %a, [2 x i7] %b, i[[#SBITS]] zeroext 0, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]], i[[#SBITS]]* %labelreturn)
  ; TLS_ABI: [[RE:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; TLS_ABI: [[RS0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] [[RE]], 0
  ; TLS_ABI: [[RS1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RS0]], i[[#SBITS]] [[RE]], 1
  ; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } [[RS1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]

  %r = call {i1, i7} @custom_cb({i1, i7} ({i32, i1}, [2 x i7])* @cb, {i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @custom_cb({i1, i7} ({i32, i1}, [2 x i7])* %cb, {i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: define { i1, i7 } @custom_cb({ i1, i7 } ({ i32, i1 }, [2 x i7])* %cb, { i32, i1 } %a, [2 x i7] %b)

  %r = call {i1, i7} %cb({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r
}

define {i1, i7} @cb({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: define { i1, i7 } @"dfs$cb"({ i32, i1 } %a, [2 x i7] %b)
  ; TLS_ABI: [[BL:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; TLS_ABI: [[AL:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: [[AL1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[AL]], 1
  ; TLS_ABI: [[BL0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[BL]], 0
  ; TLS_ABI: [[RL0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } zeroinitializer, i[[#SBITS]] [[AL1]], 0
  ; TLS_ABI: [[RL:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RL0]], i[[#SBITS]] [[BL0]], 1
  ; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } [[RL]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]

  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7}  ({i32, i1}, [2 x i7])* @ret_custom() {
  ; TLS_ABI: @"dfs$ret_custom"
  ; TLS_ABI: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  ; TLS_ABI: ret {{.*}} @"dfsw$custom_with_ret"
  ret {i1, i7}  ({i32, i1}, [2 x i7])* @custom_with_ret
}

; TLS_ABI: define linkonce_odr { i1, i7 } @"dfsw$custom_cb"({ i1, i7 } ({ i32, i1 }, [2 x i7])* %0, { i32, i1 } %1, [2 x i7] %2) {
; TLS_ABI: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
; COMM: TODO simplify the expression [[#mul(2,SBYTES) + max(SBYTES,2)]] to
; COMM: [[#mul(3,SBYTES)]], if shadow-tls-alignment is updated to match shadow
; COMM: width bytes.
; TLS_ABI: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES) + max(SBYTES,2)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
; TLS_ABI: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
; TLS_ABI: [[CB:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([100 x i64]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
; TLS_ABI: [[CAST:%.*]] = bitcast { i1, i7 } ({ i32, i1 }, [2 x i7])* %0 to i8*
; TLS_ABI: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
; TLS_ABI: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
; TLS_ABI: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
; TLS_ABI: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
; TLS_ABI: [[R:%.*]]  = call { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)* @"dfst0$custom_cb", i8* [[CAST]], { i32, i1 } %1, [2 x i7] %2, i[[#SBITS]] zeroext [[CB]], i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]], i[[#SBITS]]* %labelreturn)
; TLS_ABI: [[RE:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
; TLS_ABI: [[RS0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] [[RE]], 0
; TLS_ABI: [[RS1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RS0]], i[[#SBITS]] [[RE]], 1
; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } [[RS1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]


define {i1, i7} @custom_with_ret({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: define linkonce_odr { i1, i7 } @"dfsw$custom_with_ret"({ i32, i1 } %0, [2 x i7] %1)
  ; TLS_ABI: %labelreturn = alloca i[[#SBITS]], align [[#SBYTES]]
  ; TLS_ABI: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; TLS_ABI: [[R:%.*]] = call { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 } %0, [2 x i7] %1, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]], i[[#SBITS]]* %labelreturn)
  ; TLS_ABI: [[RE:%.*]] = load i[[#SBITS]], i[[#SBITS]]* %labelreturn, align [[#SBYTES]]
  ; TLS_ABI: [[RS0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] [[RE]], 0
  ; TLS_ABI: [[RS1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[RS0]], i[[#SBITS]] [[RE]], 1
  ; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } [[RS1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: ret { i1, i7 } [[R]]
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define void @custom_without_ret({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: define linkonce_odr void @"dfsw$custom_without_ret"({ i32, i1 } %0, [2 x i7] %1)
  ; TLS_ABI: [[B:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i[[#SBITS]] [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i[[#SBITS]] [[B0]], [[B1]]
  ; TLS_ABI: call void @__dfsw_custom_without_ret({ i32, i1 } %0, [2 x i7] %1, i[[#SBITS]] zeroext [[A01]], i[[#SBITS]] zeroext [[B01]])
  ; TLS_ABI: ret
  ret void
}

define void @custom_varg({i32, i1} %a, ...) {
  ; TLS_ABI: define linkonce_odr void @"dfsw$custom_varg"({ i32, i1 } %0, ...)
  ; TLS_ABI: call void @__dfsan_vararg_wrapper
  ; TLS_ABI: unreachable
  ret void
}

; TLS_ABI: declare { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)
; TLS_ABI: declare void @__dfsw_custom_without_ret({ i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]])
; TLS_ABI: declare void @__dfsw_custom_varg({ i32, i1 }, i[[#SBITS]], i[[#SBITS]]*, ...)

; TLS_ABI: declare { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)*, i8*, { i32, i1 }, [2 x i7], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]], i[[#SBITS]]*)

; TLS_ABI: define linkonce_odr { i1, i7 } @"dfst0$custom_cb"({ i1, i7 } ({ i32, i1 }, [2 x i7])* %0, { i32, i1 } %1, [2 x i7] %2, i[[#SBITS]] %3, i[[#SBITS]] %4, i[[#SBITS]]* %5) {
; TLS_ABI: [[A0:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } undef, i[[#SBITS]] %3, 0
; TLS_ABI: [[A1:%.*]] = insertvalue { i[[#SBITS]], i[[#SBITS]] } [[A0]], i[[#SBITS]] %3, 1
; TLS_ABI: [[B0:%.*]] = insertvalue [2 x i[[#SBITS]]] undef, i[[#SBITS]] %4, 0
; TLS_ABI: [[B1:%.*]] = insertvalue [2 x i[[#SBITS]]] [[B0]], i[[#SBITS]] %4, 1
; TLS_ABI: store { i[[#SBITS]], i[[#SBITS]] } [[A1]], { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN:2]]
; TLS_ABI: store [2 x i[[#SBITS]]] [[B1]], [2 x i[[#SBITS]]]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 [[#mul(2,SBYTES)]]) to [2 x i[[#SBITS]]]*), align [[ALIGN]]
; TLS_ABI: [[R:%.*]] = call { i1, i7 } %0({ i32, i1 } %1, [2 x i7] %2)
; TLS_ABI: %_dfsret = load { i[[#SBITS]], i[[#SBITS]] }, { i[[#SBITS]], i[[#SBITS]] }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i[[#SBITS]], i[[#SBITS]] }*), align [[ALIGN]]
; TLS_ABI: [[RE0:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } %_dfsret, 0
; TLS_ABI: [[RE1:%.*]] = extractvalue { i[[#SBITS]], i[[#SBITS]] } %_dfsret, 1
; TLS_ABI: [[RE01:%.*]] = or i[[#SBITS]] [[RE0]], [[RE1]]
; TLS_ABI: store i[[#SBITS]] [[RE01]], i[[#SBITS]]* %5, align [[#SBYTES]]
; TLS_ABI: ret { i1, i7 } [[R]]
