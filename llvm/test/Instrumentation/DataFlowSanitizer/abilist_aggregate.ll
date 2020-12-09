; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefix=TLS_ABI
; RUN: opt < %s -dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefix=LEGACY
; RUN: opt < %s -dfsan -dfsan-args-abi -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck %s --check-prefix=ARGS_ABI
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

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
  ; TLS_ABI: @"dfs$call_functional"
  ; TLS_ABI: [[B:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i16, i16 } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i16, i16 } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i16 [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i16] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i16] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i16 [[B0]], [[B1]]
  ; TLS_ABI: [[U:%.*]] = or i16 [[A01]], [[B01]]
  ; TLS_ABI: [[R0:%.*]] = insertvalue { i16, i16 } undef, i16 [[U]], 0
  ; TLS_ABI: [[R1:%.*]] = insertvalue { i16, i16 } [[R0]], i16 [[U]], 1
  ; TLS_ABI: store { i16, i16 } [[R1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
  
  ; LEGACY: @"dfs$call_functional"
  ; LEGACY: [[B:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align [[ALIGN:2]]
  ; LEGACY: [[A:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; LEGACY: [[U:%.*]] = call zeroext i16 @__dfsan_union(i16 zeroext [[A]], i16 zeroext [[B]])
  ; LEGACY: [[PH:%.*]] = phi i16 [ [[U]], {{.*}} ], [ [[A]], {{.*}} ]
  ; LEGACY: store i16 [[PH]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]

  ; ARGS_ABI: @"dfs$call_functional"
  ; ARGS_ABI: [[U:%.*]]  = call zeroext i16 @__dfsan_union(i16 zeroext %2, i16 zeroext %3)
  ; ARGS_ABI: [[PH:%.*]] = phi i16 [ %7, {{.*}} ], [ %2, {{.*}} ]
  ; ARGS_ABI: [[R0:%.*]] = insertvalue { { i1, i7 }, i16 } undef, { i1, i7 } %r, 0
  ; ARGS_ABI: [[R1:%.*]] = insertvalue { { i1, i7 }, i16 } [[R0]], i16 [[PH]], 1
  ; ARGS_ABI: ret { { i1, i7 }, i16 } [[R1]]
  
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
  ; TLS_ABI: store { i16, i16 } zeroinitializer, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align 2
  
  ; ARGS_ABI: @"dfs$call_discard"
  ; ARGS_ABI: %r = call { i1, i7 } @discard({ i32, i1 } %0, [2 x i7] %1)
  ; ARGS_ABI: [[R0:%.*]] = insertvalue { { i1, i7 }, i16 } undef, { i1, i7 } %r, 0
  ; ARGS_ABI: [[R1:%.*]] = insertvalue { { i1, i7 }, i16 } [[R0]], i16 0, 1
  ; ARGS_ABI: ret { { i1, i7 }, i16 } [[R1]]
  
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
  ; TLS_ABI: store { i16, i16 } zeroinitializer, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align 2
  
  ; ARGS_ABI: @"dfs$call_uninstrumented"
  ; ARGS_ABI: call void @__dfsan_unimplemented
  ; ARGS_ABI: %r = call { i1, i7 } @uninstrumented({ i32, i1 } %0, [2 x i7] %1)
  ; ARGS_ABI: [[R0:%.*]] = insertvalue { { i1, i7 }, i16 } undef, { i1, i7 } %r, 0
  ; ARGS_ABI: [[R1:%.*]] = insertvalue { { i1, i7 }, i16 } [[R0]], i16 0, 1
  ; ARGS_ABI: ret { { i1, i7 }, i16 } [[R1]]
  
  %r = call {i1, i7} @uninstrumented({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r  
}

define {i1, i7} @call_custom_with_ret({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: @"dfs$call_custom_with_ret"
  ; TLS_ABI: %labelreturn = alloca i16, align 2
  ; TLS_ABI: [[B:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i16, i16 } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i16, i16 } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i16 [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i16] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i16] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i16 [[B0]], [[B1]]
  ; TLS_ABI: [[R:%.*]] = call { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 } %a, [2 x i7] %b, i16 zeroext [[A01]], i16 zeroext [[B01]], i16* %labelreturn)
  ; TLS_ABI: [[RE:%.*]] = load i16, i16* %labelreturn, align [[ALIGN]]
  ; TLS_ABI: [[RS0:%.*]] = insertvalue { i16, i16 } undef, i16 [[RE]], 0
  ; TLS_ABI: [[RS1:%.*]] = insertvalue { i16, i16 } [[RS0]], i16 [[RE]], 1
  ; TLS_ABI: store { i16, i16 } [[RS1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: ret { i1, i7 } [[R]]
  
  %r = call {i1, i7} @custom_with_ret({i32, i1} %a, [2 x i7] %b)
  ret {i1, i7} %r  
}

define void @call_custom_without_ret({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: @"dfs$call_custom_without_ret"
  ; TLS_ABI: [[B:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i16, i16 } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i16, i16 } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i16 [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i16] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i16] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i16 [[B0]], [[B1]]
  ; TLS_ABI: call void @__dfsw_custom_without_ret({ i32, i1 } %a, [2 x i7] %b, i16 zeroext [[A01]], i16 zeroext [[B01]])
  
  call void @custom_without_ret({i32, i1} %a, [2 x i7] %b)
  ret void
}

define void @call_custom_varg({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: @"dfs$call_custom_varg"
  ; TLS_ABI: [[B:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN:2]]
  ; TLS_ABI: %labelva = alloca [1 x i16], align [[ALIGN]]
  ; TLS_ABI: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i16, i16 } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i16, i16 } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i16 [[A0]], [[A1]]
  ; TLS_ABI: [[V0:%.*]] = getelementptr inbounds [1 x i16], [1 x i16]* %labelva, i32 0, i32 0
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i16] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i16] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i16 [[B0]], [[B1]]
  ; TLS_ABI: store i16 [[B01]], i16* [[V0]], align 2
  ; TLS_ABI: [[V:%.*]] = getelementptr inbounds [1 x i16], [1 x i16]* %labelva, i32 0, i32 0
  ; TLS_ABI: call void ({ i32, i1 }, i16, i16*, ...) @__dfsw_custom_varg({ i32, i1 } %a, i16 zeroext [[A01]], i16* [[V]], [2 x i7] %b)

  call void ({i32, i1}, ...) @custom_varg({i32, i1} %a, [2 x i7] %b)
  ret void
}

define {i1, i7} @call_custom_cb({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: define { i1, i7 } @"dfs$call_custom_cb"({ i32, i1 } %a, [2 x i7] %b) {
  ; TLS_ABI: %labelreturn = alloca i16, align 2
  ; TLS_ABI: [[B:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i16, i16 } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i16, i16 } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i16 [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i16] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i16] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i16 [[B0]], [[B1]]  
  ; TLS_ABI: [[R:%.*]]  = call { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i16, i16, i16*)* @"dfst0$custom_cb", i8* bitcast ({ i1, i7 } ({ i32, i1 }, [2 x i7])* @"dfs$cb" to i8*), { i32, i1 } %a, [2 x i7] %b, i16 zeroext 0, i16 zeroext [[A01]], i16 zeroext [[B01]], i16* %labelreturn)
  ; TLS_ABI: [[RE:%.*]] = load i16, i16* %labelreturn, align [[ALIGN]]
  ; TLS_ABI: [[RS0:%.*]] = insertvalue { i16, i16 } undef, i16 [[RE]], 0
  ; TLS_ABI: [[RS1:%.*]] = insertvalue { i16, i16 } [[RS0]], i16 [[RE]], 1
  ; TLS_ABI: store { i16, i16 } [[RS1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]

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
  ; TLS_ABI: [[BL:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN:2]]
  ; TLS_ABI: [[AL:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: [[AL1:%.*]] = extractvalue { i16, i16 } [[AL]], 1
  ; TLS_ABI: [[BL0:%.*]] = extractvalue [2 x i16] [[BL]], 0
  ; TLS_ABI: [[RL0:%.*]] = insertvalue { i16, i16 } zeroinitializer, i16 [[AL1]], 0
  ; TLS_ABI: [[RL:%.*]] = insertvalue { i16, i16 } [[RL0]], i16 [[BL0]], 1
  ; TLS_ABI: store { i16, i16 } [[RL]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]

  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define {i1, i7}  ({i32, i1}, [2 x i7])* @ret_custom() {
  ; TLS_ABI: @"dfs$ret_custom"
  ; TLS_ABI: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
  ; TLS_ABI: ret {{.*}} @"dfsw$custom_with_ret"
  ret {i1, i7}  ({i32, i1}, [2 x i7])* @custom_with_ret
}

; TLS_ABI: define linkonce_odr { i1, i7 } @"dfsw$custom_cb"({ i1, i7 } ({ i32, i1 }, [2 x i7])* %0, { i32, i1 } %1, [2 x i7] %2) {
; TLS_ABI: %labelreturn = alloca i16, align 2
; TLS_ABI: [[B:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 6) to [2 x i16]*), align [[ALIGN:2]]
; TLS_ABI: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to { i16, i16 }*), align [[ALIGN]]
; TLS_ABI: [[CB:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
; TLS_ABI: [[CAST:%.*]] = bitcast { i1, i7 } ({ i32, i1 }, [2 x i7])* %0 to i8*
; TLS_ABI: [[A0:%.*]] = extractvalue { i16, i16 } [[A]], 0
; TLS_ABI: [[A1:%.*]] = extractvalue { i16, i16 } [[A]], 1
; TLS_ABI: [[A01:%.*]] = or i16 [[A0]], [[A1]]
; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i16] [[B]], 0
; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i16] [[B]], 1
; TLS_ABI: [[B01:%.*]] = or i16 [[B0]], [[B1]]  
; TLS_ABI: [[R:%.*]]  = call { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i16, i16, i16*)* @"dfst0$custom_cb", i8* [[CAST]], { i32, i1 } %1, [2 x i7] %2, i16 zeroext [[CB]], i16 zeroext [[A01]], i16 zeroext [[B01]], i16* %labelreturn)
; TLS_ABI: [[RE:%.*]] = load i16, i16* %labelreturn, align [[ALIGN]]
; TLS_ABI: [[RS0:%.*]] = insertvalue { i16, i16 } undef, i16 [[RE]], 0
; TLS_ABI: [[RS1:%.*]] = insertvalue { i16, i16 } [[RS0]], i16 [[RE]], 1
; TLS_ABI: store { i16, i16 } [[RS1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
 

define {i1, i7} @custom_with_ret({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: define linkonce_odr { i1, i7 } @"dfsw$custom_with_ret"({ i32, i1 } %0, [2 x i7] %1)
  ; TLS_ABI: %labelreturn = alloca i16, align 2
  ; TLS_ABI: [[B:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i16, i16 } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i16, i16 } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i16 [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i16] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i16] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i16 [[B0]], [[B1]]
  ; TLS_ABI: [[R:%.*]] = call { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 } %0, [2 x i7] %1, i16 zeroext [[A01]], i16 zeroext [[B01]], i16* %labelreturn)
  ; TLS_ABI: [[RE:%.*]] = load i16, i16* %labelreturn, align 2
  ; TLS_ABI: [[RS0:%.*]] = insertvalue { i16, i16 } undef, i16 [[RE]], 0
  ; TLS_ABI: [[RS1:%.*]] = insertvalue { i16, i16 } [[RS0]], i16 [[RE]], 1
  ; TLS_ABI: store { i16, i16 } [[RS1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: ret { i1, i7 } [[R]]
  %a1 = extractvalue {i32, i1} %a, 1
  %b0 = extractvalue [2 x i7] %b, 0
  %r0 = insertvalue {i1, i7} undef, i1 %a1, 0
  %r1 = insertvalue {i1, i7} %r0, i7 %b0, 1
  ret {i1, i7} %r1
}

define void @custom_without_ret({i32, i1} %a, [2 x i7] %b) {
  ; TLS_ABI: define linkonce_odr void @"dfsw$custom_without_ret"({ i32, i1 } %0, [2 x i7] %1)
  ; TLS_ABI: [[B:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN:2]]
  ; TLS_ABI: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; TLS_ABI: [[A0:%.*]] = extractvalue { i16, i16 } [[A]], 0
  ; TLS_ABI: [[A1:%.*]] = extractvalue { i16, i16 } [[A]], 1
  ; TLS_ABI: [[A01:%.*]] = or i16 [[A0]], [[A1]]
  ; TLS_ABI: [[B0:%.*]] = extractvalue [2 x i16] [[B]], 0
  ; TLS_ABI: [[B1:%.*]] = extractvalue [2 x i16] [[B]], 1
  ; TLS_ABI: [[B01:%.*]] = or i16 [[B0]], [[B1]]
  ; TLS_ABI: call void @__dfsw_custom_without_ret({ i32, i1 } %0, [2 x i7] %1, i16 zeroext [[A01]], i16 zeroext [[B01]])
  ; TLS_ABI: ret
  ret void
}

define void @custom_varg({i32, i1} %a, ...) {
  ; TLS_ABI: define linkonce_odr void @"dfsw$custom_varg"({ i32, i1 } %0, ...)
  ; TLS_ABI: call void @__dfsan_vararg_wrapper
  ; TLS_ABI: unreachable
  ret void
}

; TLS_ABI: declare { i1, i7 } @__dfsw_custom_with_ret({ i32, i1 }, [2 x i7], i16, i16, i16*)
; TLS_ABI: declare void @__dfsw_custom_without_ret({ i32, i1 }, [2 x i7], i16, i16)
; TLS_ABI: declare void @__dfsw_custom_varg({ i32, i1 }, i16, i16*, ...)

; TLS_ABI: declare { i1, i7 } @__dfsw_custom_cb({ i1, i7 } ({ i1, i7 } ({ i32, i1 }, [2 x i7])*, { i32, i1 }, [2 x i7], i16, i16, i16*)*, i8*, { i32, i1 }, [2 x i7], i16, i16, i16, i16*)

; TLS_ABI: define linkonce_odr { i1, i7 } @"dfst0$custom_cb"({ i1, i7 } ({ i32, i1 }, [2 x i7])* %0, { i32, i1 } %1, [2 x i7] %2, i16 %3, i16 %4, i16* %5) {
; TLS_ABI: [[A0:%.*]] = insertvalue { i16, i16 } undef, i16 %3, 0
; TLS_ABI: [[A1:%.*]] = insertvalue { i16, i16 } [[A0]], i16 %3, 1
; TLS_ABI: [[B0:%.*]] = insertvalue [2 x i16] undef, i16 %4, 0
; TLS_ABI: [[B1:%.*]] = insertvalue [2 x i16] [[B0]], i16 %4, 1
; TLS_ABI: store { i16, i16 } [[A1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN:2]]
; TLS_ABI: store [2 x i16] [[B1]], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to [2 x i16]*), align [[ALIGN]]
; TLS_ABI: [[R:%.*]] = call { i1, i7 } %0({ i32, i1 } %1, [2 x i7] %2)
; TLS_ABI: %_dfsret = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
; TLS_ABI: [[RE0:%.*]] = extractvalue { i16, i16 } %_dfsret, 0
; TLS_ABI: [[RE1:%.*]] = extractvalue { i16, i16 } %_dfsret, 1
; TLS_ABI: [[RE01:%.*]] = or i16 [[RE0]], [[RE1]]
; TLS_ABI: store i16 [[RE01]], i16* %5, align [[ALIGN]]
; TLS_ABI: ret { i1, i7 } [[R]]
