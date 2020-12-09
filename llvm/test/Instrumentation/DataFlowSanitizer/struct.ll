; RUN: opt < %s -dfsan -S | FileCheck %s --check-prefix=LEGACY
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-event-callbacks=true -S | FileCheck %s --check-prefix=EVENT_CALLBACKS
; RUN: opt < %s -dfsan -dfsan-args-abi -S | FileCheck %s --check-prefix=ARGS_ABI
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefix=FAST16
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s --check-prefix=NO_COMBINE_LOAD_PTR
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-store=true -S | FileCheck %s --check-prefix=COMBINE_STORE_PTR
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-track-select-control-flow=false -S | FileCheck %s --check-prefix=NO_SELECT_CONTROL
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-debug-nonzero-labels -S | FileCheck %s --check-prefix=DEBUG_NONZERO_LABELS
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define {i8*, i32} @pass_struct({i8*, i32} %s) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$pass_struct"
  ; NO_COMBINE_LOAD_PTR: [[L:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store { i16, i16 } [[L]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]

  ; ARGS_ABI: @"dfs$pass_struct"
  ; ARGS_ABI: ret { { i8*, i32 }, i16 }
  
  ; DEBUG_NONZERO_LABELS: @"dfs$pass_struct"
  ; DEBUG_NONZERO_LABELS: [[L:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN:2]]
  ; DEBUG_NONZERO_LABELS: [[L0:%.*]] = extractvalue { i16, i16 } [[L]], 0
  ; DEBUG_NONZERO_LABELS: [[L1:%.*]] = extractvalue { i16, i16 } [[L]], 1
  ; DEBUG_NONZERO_LABELS: [[L01:%.*]] = or i16 [[L0]], [[L1]]
  ; DEBUG_NONZERO_LABELS: {{.*}} = icmp ne i16 [[L01]], 0
  ; DEBUG_NONZERO_LABELS: call void @__dfsan_nonzero_label()
  ; DEBUG_NONZERO_LABELS: store { i16, i16 } [[L]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
  
  ret {i8*, i32} %s
}

%StructOfAggr = type {i8*, [4 x i2], <4 x i3>, {i1, i1}}

define %StructOfAggr @pass_struct_of_aggregate(%StructOfAggr %s) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$pass_struct_of_aggregate"
  ; NO_COMBINE_LOAD_PTR: %1 = load { i16, [4 x i16], i16, { i16, i16 } }, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store { i16, [4 x i16], i16, { i16, i16 } } %1, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN]]

  ; ARGS_ABI: @"dfs$pass_struct_of_aggregate"
  ; ARGS_ABI: ret { %StructOfAggr, i16 }
  ret %StructOfAggr %s
}

define {} @load_empty_struct({}* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_empty_struct"
  ; NO_COMBINE_LOAD_PTR: store {} zeroinitializer, {}* bitcast ([100 x i64]* @__dfsan_retval_tls to {}*), align 2

  %a = load {}, {}* %p
  ret {} %a
}

@Y = constant {i1, i32} {i1 1, i32 1}

define {i1, i32} @load_global_struct() {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_global_struct"
  ; NO_COMBINE_LOAD_PTR: store { i16, i16 } zeroinitializer, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align 2

  %a = load {i1, i32}, {i1, i32}* @Y
  ret {i1, i32} %a
}

define {i1, i32} @select_struct(i1 %c, {i1, i32} %a, {i1, i32} %b) {
  ; NO_SELECT_CONTROL: @"dfs$select_struct"
  ; NO_SELECT_CONTROL: [[B:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 6) to { i16, i16 }*), align [[ALIGN:2]]
  ; NO_SELECT_CONTROL: [[A:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to { i16, i16 }*), align [[ALIGN]]
  ; NO_SELECT_CONTROL: [[C:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; NO_SELECT_CONTROL: [[S:%.*]] = select i1 %c, { i16, i16 } [[A]], { i16, i16 } [[B]]
  ; NO_SELECT_CONTROL: store { i16, i16 } [[S]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]

  ; FAST16: @"dfs$select_struct"
  ; FAST16: [[B_S:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 6) to { i16, i16 }*), align [[ALIGN:2]]
  ; FAST16: [[A_S:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to { i16, i16 }*), align [[ALIGN]]
  ; FAST16: [[C_S:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; FAST16: [[S_S:%.*]] = select i1 %c, { i16, i16 } [[A_S]], { i16, i16 } [[B_S]]
  ; FAST16: [[S0_S:%.*]] = extractvalue { i16, i16 } [[S_S]], 0
  ; FAST16: [[S1_S:%.*]] = extractvalue { i16, i16 } [[S_S]], 1
  ; FAST16: [[S01_S:%.*]] = or i16 [[S0_S]], [[S1_S]]
  ; FAST16: [[CS_S:%.*]] = or i16 [[C_S]], [[S01_S]]
  ; FAST16: [[S1:%.*]] = insertvalue { i16, i16 } undef, i16 [[CS_S]], 0
  ; FAST16: [[S2:%.*]] = insertvalue { i16, i16 } [[S1]], i16 [[CS_S]], 1
  ; FAST16: store { i16, i16 } [[S2]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]

  ; LEGACY: @"dfs$select_struct"
  ; LEGACY: [[U:%.*]] = call zeroext i16 @__dfsan_union
  ; LEGACY: [[P:%.*]] = phi i16 [ [[U]],
  ; LEGACY: store i16 [[P]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2

  %s = select i1 %c, {i1, i32} %a, {i1, i32} %b
  ret {i1, i32} %s
}

define { i32, i32 } @asm_struct(i32 %0, i32 %1) {
  ; FAST16: @"dfs$asm_struct"
  ; FAST16: [[E1:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align [[ALIGN:2]]
  ; FAST16: [[E0:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; FAST16: [[E01:%.*]] = or i16 [[E0]], [[E1]]
  ; FAST16: [[S0:%.*]] = insertvalue { i16, i16 } undef, i16 [[E01]], 0
  ; FAST16: [[S1:%.*]] = insertvalue { i16, i16 } [[S0]], i16 [[E01]], 1
  ; FAST16: store { i16, i16 } [[S1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]

  ; LEGACY: @"dfs$asm_struct"
  ; LEGACY: [[E1:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align [[ALIGN:2]]
  ; LEGACY: [[E0:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; LEGACY: [[E01:%.*]] = call zeroext i16 @__dfsan_union(i16 zeroext [[E0]], i16 zeroext [[E1]])
  ; LEGACY: [[P:%.*]] = phi i16 [ [[E01]], {{.*}} ], [ [[E0]], {{.*}} ]
  ; LEGACY: store i16 [[P]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  
entry:
  %a = call { i32, i32 } asm "", "=r,=r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
  ret { i32, i32 } %a
}

define {i32, i32} @const_struct() {
  ; FAST16: @"dfs$const_struct"
  ; FAST16: store { i16, i16 } zeroinitializer, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align 2
  
  ; LEGACY: @"dfs$const_struct"
  ; LEGACY: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
  ret {i32, i32} { i32 42, i32 11 }
}

define i1 @extract_struct({i1, i5} %s) {
  ; FAST16: @"dfs$extract_struct"
  ; FAST16: [[SM:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN:2]]
  ; FAST16: [[EM:%.*]] = extractvalue { i16, i16 } [[SM]], 0
  ; FAST16: store i16 [[EM]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  
  ; LEGACY: @"dfs$extract_struct"
  ; LEGACY: [[SM:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; LEGACY: store i16 [[SM]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  %e2 = extractvalue {i1, i5} %s, 0
  ret i1 %e2
}

define {i1, i5} @insert_struct({i1, i5} %s, i5 %e1) {
  ; FAST16: @"dfs$insert_struct"
  ; FAST16: [[EM:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 4) to i16*), align [[ALIGN:2]]
  ; FAST16: [[SM:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; FAST16: [[SM1:%.*]] = insertvalue { i16, i16 } [[SM]], i16 [[EM]], 1
  ; FAST16: store { i16, i16 } [[SM1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
  
  ; LEGACY: @"dfs$insert_struct"
  ; LEGACY: [[EM:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align [[ALIGN:2]]
  ; LEGACY: [[SM:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; LEGACY: [[U:%.*]] = call zeroext i16 @__dfsan_union(i16 zeroext [[SM]], i16 zeroext [[EM]])
  ; LEGACY: [[P:%.*]] = phi i16 [ [[U]], {{.*}} ], [ [[SM]], {{.*}} ]
  ; LEGACY: store i16 [[P]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  %s1 = insertvalue {i1, i5} %s, i5 %e1, 1
  ret {i1, i5} %s1
}

define {i1, i1} @load_struct({i1, i1}* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_struct"
  ; NO_COMBINE_LOAD_PTR: [[OL:%.*]] = or i16
  ; NO_COMBINE_LOAD_PTR: [[S0:%.*]] = insertvalue { i16, i16 } undef, i16 [[OL]], 0
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue { i16, i16 } [[S0]], i16 [[OL]], 1
  ; NO_COMBINE_LOAD_PTR: store { i16, i16 } [[S1]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align 2
  
  ; EVENT_CALLBACKS: @"dfs$load_struct"
  ; EVENT_CALLBACKS: [[OL0:%.*]] = or i16
  ; EVENT_CALLBACKS: [[OL1:%.*]] = or i16 [[OL0]],
  ; EVENT_CALLBACKS: [[S0:%.*]] = insertvalue { i16, i16 } undef, i16 [[OL1]], 0
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i16 [[OL1]]
  
  %s = load {i1, i1}, {i1, i1}* %p
  ret {i1, i1} %s
}

define void @store_struct({i1, i1}* %p, {i1, i1} %s) {
  ; FAST16: @"dfs$store_struct"
  ; FAST16: [[S:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to { i16, i16 }*), align [[ALIGN:2]]
  ; FAST16: [[E0:%.*]] = extractvalue { i16, i16 } [[S]], 0
  ; FAST16: [[E1:%.*]] = extractvalue { i16, i16 } [[S]], 1
  ; FAST16: [[E:%.*]] = or i16 [[E0]], [[E1]]
  ; FAST16: [[P0:%.*]] = getelementptr i16, i16* [[P:%.*]], i32 0
  ; FAST16: store i16 [[E]], i16* [[P0]], align [[ALIGN]]
  ; FAST16: [[P1:%.*]] = getelementptr i16, i16* [[P]], i32 1
  ; FAST16: store i16 [[E]], i16* [[P1]], align [[ALIGN]]
  
  ; EVENT_CALLBACKS: @"dfs$store_struct"
  ; EVENT_CALLBACKS: [[OL:%.*]] = or i16
  ; EVENT_CALLBACKS: call void @__dfsan_store_callback(i16 [[OL]]
  
  ; COMBINE_STORE_PTR: @"dfs$store_struct"
  ; COMBINE_STORE_PTR: [[PL:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; COMBINE_STORE_PTR: [[SL:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to { i16, i16 }*), align [[ALIGN]]
  ; COMBINE_STORE_PTR: [[SL0:%.*]] = extractvalue { i16, i16 } [[SL]], 0
  ; COMBINE_STORE_PTR: [[SL1:%.*]] = extractvalue { i16, i16 } [[SL]], 1
  ; COMBINE_STORE_PTR: [[SL01:%.*]] = or i16 [[SL0]], [[SL1]]
  ; COMBINE_STORE_PTR: [[E:%.*]] = or i16 [[SL01]], [[PL]]
  ; COMBINE_STORE_PTR: [[P0:%.*]] = getelementptr i16, i16* [[P:%.*]], i32 0
  ; COMBINE_STORE_PTR: store i16 [[E]], i16* [[P0]], align [[ALIGN]]
  ; COMBINE_STORE_PTR: [[P1:%.*]] = getelementptr i16, i16* [[P]], i32 1
  ; COMBINE_STORE_PTR: store i16 [[E]], i16* [[P1]], align [[ALIGN]]
  
  store {i1, i1} %s, {i1, i1}* %p
  ret void
}

define i2 @extract_struct_of_aggregate11(%StructOfAggr %s) {
  ; FAST16: @"dfs$extract_struct_of_aggregate11"
  ; FAST16: [[E:%.*]] = load { i16, [4 x i16], i16, { i16, i16 } }, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN:2]]
  ; FAST16: [[E11:%.*]] = extractvalue { i16, [4 x i16], i16, { i16, i16 } } [[E]], 1, 1
  ; FAST16: store i16 [[E11]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]

  %e11 = extractvalue %StructOfAggr %s, 1, 1
  ret i2 %e11
}

define [4 x i2] @extract_struct_of_aggregate1(%StructOfAggr %s) {
  ; FAST16: @"dfs$extract_struct_of_aggregate1"
  ; FAST16: [[E:%.*]] = load { i16, [4 x i16], i16, { i16, i16 } }, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN:2]]
  ; FAST16: [[E1:%.*]] = extractvalue { i16, [4 x i16], i16, { i16, i16 } } [[E]], 1
  ; FAST16: store [4 x i16] [[E1]], [4 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x i16]*), align [[ALIGN]]
  %e1 = extractvalue %StructOfAggr %s, 1
  ret [4 x i2] %e1
}

define <4 x i3> @extract_struct_of_aggregate2(%StructOfAggr %s) {
  ; FAST16: @"dfs$extract_struct_of_aggregate2"
  ; FAST16: [[E:%.*]] = load { i16, [4 x i16], i16, { i16, i16 } }, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN:2]]
  ; FAST16: [[E2:%.*]] = extractvalue { i16, [4 x i16], i16, { i16, i16 } } [[E]], 2
  ; FAST16: store i16 [[E2]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  %e2 = extractvalue %StructOfAggr %s, 2
  ret <4 x i3> %e2
}

define { i1, i1 } @extract_struct_of_aggregate3(%StructOfAggr %s) {
  ; FAST16: @"dfs$extract_struct_of_aggregate3"
  ; FAST16: [[E:%.*]] = load { i16, [4 x i16], i16, { i16, i16 } }, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN:2]]
  ; FAST16: [[E3:%.*]] = extractvalue { i16, [4 x i16], i16, { i16, i16 } } [[E]], 3
  ; FAST16: store { i16, i16 } [[E3]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
  %e3 = extractvalue %StructOfAggr %s, 3
  ret { i1, i1 } %e3
}

define i1 @extract_struct_of_aggregate31(%StructOfAggr %s) {
  ; FAST16: @"dfs$extract_struct_of_aggregate31"
  ; FAST16: [[E:%.*]] = load { i16, [4 x i16], i16, { i16, i16 } }, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN:2]]
  ; FAST16: [[E31:%.*]] = extractvalue { i16, [4 x i16], i16, { i16, i16 } } [[E]], 3, 1
  ; FAST16: store i16 [[E31]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  %e31 = extractvalue %StructOfAggr %s, 3, 1
  ret i1 %e31
}

define %StructOfAggr @insert_struct_of_aggregate11(%StructOfAggr %s, i2 %e11) {
  ; FAST16: @"dfs$insert_struct_of_aggregate11"
  ; FAST16: [[E11:%.*]]  = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 16) to i16*), align [[ALIGN:2]]
  ; FAST16: [[S:%.*]]  = load { i16, [4 x i16], i16, { i16, i16 } }, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN]]
  ; FAST16: [[S1:%.*]]  = insertvalue { i16, [4 x i16], i16, { i16, i16 } } [[S]], i16 [[E11]], 1, 1
  ; FAST16: store { i16, [4 x i16], i16, { i16, i16 } } [[S1]], { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN]]

  %s1 = insertvalue %StructOfAggr %s, i2 %e11, 1, 1
  ret %StructOfAggr %s1
}

define {i8*, i32} @call_struct({i8*, i32} %s) {
  ; FAST16: @"dfs$call_struct"
  ; FAST16: [[S:%.*]] = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN:2]]
  ; FAST16: store { i16, i16 } [[S]], { i16, i16 }* bitcast ([100 x i64]* @__dfsan_arg_tls to { i16, i16 }*), align [[ALIGN]]
  ; FAST16: %_dfsret = load { i16, i16 }, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]
  ; FAST16: store { i16, i16 } %_dfsret, { i16, i16 }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, i16 }*), align [[ALIGN]]

  %r = call {i8*, i32} @pass_struct({i8*, i32} %s)
  ret {i8*, i32} %r
}

declare %StructOfAggr @fun_with_many_aggr_args(<2 x i7> %v, [2 x i5] %a, {i3, i3} %s)

define %StructOfAggr @call_many_aggr_args(<2 x i7> %v, [2 x i5] %a, {i3, i3} %s) {
  ; FAST16: @"dfs$call_many_aggr_args"
  ; FAST16: [[S:%.*]] = load { i16, i16 }, { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 6) to { i16, i16 }*), align [[ALIGN:2]]
  ; FAST16: [[A:%.*]] = load [2 x i16], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to [2 x i16]*), align [[ALIGN]]
  ; FAST16: [[V:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; FAST16: store i16 [[V]], i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; FAST16: store [2 x i16] [[A]], [2 x i16]* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to [2 x i16]*), align [[ALIGN]]
  ; FAST16: store { i16, i16 } [[S]], { i16, i16 }* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 6) to { i16, i16 }*), align [[ALIGN]]
  ; FAST16: %_dfsret = load { i16, [4 x i16], i16, { i16, i16 } }, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN]]
  ; FAST16: store { i16, [4 x i16], i16, { i16, i16 } } %_dfsret, { i16, [4 x i16], i16, { i16, i16 } }* bitcast ([100 x i64]* @__dfsan_retval_tls to { i16, [4 x i16], i16, { i16, i16 } }*), align [[ALIGN]]

  %r = call %StructOfAggr @fun_with_many_aggr_args(<2 x i7> %v, [2 x i5] %a, {i3, i3} %s)
  ret %StructOfAggr %r
}