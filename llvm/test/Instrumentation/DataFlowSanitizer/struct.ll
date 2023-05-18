; RUN: opt < %s -passes=dfsan -dfsan-event-callbacks=true -S | FileCheck %s --check-prefixes=CHECK,EVENT_CALLBACKS
; RUN: opt < %s -passes=dfsan -S | FileCheck %s --check-prefixes=CHECK,FAST
; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s --check-prefixes=CHECK,NO_COMBINE_LOAD_PTR
; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-store=true -S | FileCheck %s --check-prefixes=CHECK,COMBINE_STORE_PTR
; RUN: opt < %s -passes=dfsan -dfsan-track-select-control-flow=false -S | FileCheck %s --check-prefixes=CHECK,NO_SELECT_CONTROL
; RUN: opt < %s -passes=dfsan -dfsan-debug-nonzero-labels -S | FileCheck %s --check-prefixes=CHECK,DEBUG_NONZERO_LABELS
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
define {ptr, i32} @pass_struct({ptr, i32} %s) {
  ; NO_COMBINE_LOAD_PTR: @pass_struct.dfsan
  ; NO_COMBINE_LOAD_PTR: [[L:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store { i8, i8 } [[L]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  ; DEBUG_NONZERO_LABELS: @pass_struct.dfsan
  ; DEBUG_NONZERO_LABELS: [[L:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; DEBUG_NONZERO_LABELS: [[L0:%.*]] = extractvalue { i8, i8 } [[L]], 0
  ; DEBUG_NONZERO_LABELS: [[L1:%.*]] = extractvalue { i8, i8 } [[L]], 1
  ; DEBUG_NONZERO_LABELS: [[L01:%.*]] = or i8 [[L0]], [[L1]]
  ; DEBUG_NONZERO_LABELS: {{.*}} = icmp ne i8 [[L01]], 0
  ; DEBUG_NONZERO_LABELS: call void @__dfsan_nonzero_label()
  ; DEBUG_NONZERO_LABELS: store { i8, i8 } [[L]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  ret {ptr, i32} %s
}

%StructOfAggr = type {ptr, [4 x i2], <4 x i3>, {i1, i1}}

define %StructOfAggr @pass_struct_of_aggregate(%StructOfAggr %s) {
  ; NO_COMBINE_LOAD_PTR: @pass_struct_of_aggregate.dfsan
  ; NO_COMBINE_LOAD_PTR: %1 = load { i8, [4 x i8], i8, { i8, i8 } }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store { i8, [4 x i8], i8, { i8, i8 } } %1, ptr @__dfsan_retval_tls, align [[ALIGN]]

  ret %StructOfAggr %s
}

define {} @load_empty_struct(ptr %p) {
  ; NO_COMBINE_LOAD_PTR: @load_empty_struct.dfsan
  ; NO_COMBINE_LOAD_PTR: store {} zeroinitializer, ptr @__dfsan_retval_tls, align 2

  %a = load {}, ptr %p
  ret {} %a
}

@Y = constant {i1, i32} {i1 1, i32 1}

define {i1, i32} @load_global_struct() {
  ; NO_COMBINE_LOAD_PTR: @load_global_struct.dfsan
  ; NO_COMBINE_LOAD_PTR: store { i8, i8 } zeroinitializer, ptr @__dfsan_retval_tls, align 2

  %a = load {i1, i32}, ptr @Y
  ret {i1, i32} %a
}

define {i1, i32} @select_struct(i1 %c, {i1, i32} %a, {i1, i32} %b) {
  ; NO_SELECT_CONTROL: @select_struct.dfsan
  ; NO_SELECT_CONTROL: [[B:%.*]] = load { i8, i8 }, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align [[ALIGN:2]]
  ; NO_SELECT_CONTROL: [[A:%.*]] = load { i8, i8 }, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]
  ; NO_SELECT_CONTROL: [[C:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; NO_SELECT_CONTROL: [[S:%.*]] = select i1 %c, { i8, i8 } [[A]], { i8, i8 } [[B]]
  ; NO_SELECT_CONTROL: store { i8, i8 } [[S]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  ; FAST: @select_struct.dfsan
  ; FAST: %[[#R:]] = load { i8, i8 }, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align [[ALIGN:2]]
  ; FAST: %[[#R+1]] = load { i8, i8 }, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]
  ; FAST: %[[#R+2]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: %[[#R+3]] = select i1 %c, { i8, i8 } %[[#R+1]], { i8, i8 } %[[#R]]
  ; FAST: %[[#R+4]] = extractvalue { i8, i8 } %[[#R+3]], 0
  ; FAST: %[[#R+5]] = extractvalue { i8, i8 } %[[#R+3]], 1
  ; FAST: %[[#R+6]] = or i8 %[[#R+4]], %[[#R+5]]
  ; FAST: %[[#R+7]] = or i8 %[[#R+2]], %[[#R+6]]
  ; FAST: %[[#R+8]] = insertvalue { i8, i8 } undef, i8 %[[#R+7]], 0
  ; FAST: %[[#R+9]] = insertvalue { i8, i8 } %[[#R+8]], i8 %[[#R+7]], 1
  ; FAST: store { i8, i8 } %[[#R+9]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  %s = select i1 %c, {i1, i32} %a, {i1, i32} %b
  ret {i1, i32} %s
}

define { i32, i32 } @asm_struct(i32 %0, i32 %1) {
  ; FAST: @asm_struct.dfsan
  ; FAST: [[E1:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN:2]]
  ; FAST: [[E0:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: [[E01:%.*]] = or i8 [[E0]], [[E1]]
  ; FAST: [[S0:%.*]] = insertvalue { i8, i8 } undef, i8 [[E01]], 0
  ; FAST: [[S1:%.*]] = insertvalue { i8, i8 } [[S0]], i8 [[E01]], 1
  ; FAST: store { i8, i8 } [[S1]], ptr @__dfsan_retval_tls, align [[ALIGN]]

entry:
  %a = call { i32, i32 } asm "", "=r,=r,r,r,~{dirflag},~{fpsr},~{flags}"(i32 %0, i32 %1)
  ret { i32, i32 } %a
}

define {i32, i32} @const_struct() {
  ; FAST: @const_struct.dfsan
  ; FAST: store { i8, i8 } zeroinitializer, ptr @__dfsan_retval_tls, align 2
  ret {i32, i32} { i32 42, i32 11 }
}

define i1 @extract_struct({i1, i5} %s) {
  ; FAST: @extract_struct.dfsan
  ; FAST: [[SM:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[EM:%.*]] = extractvalue { i8, i8 } [[SM]], 0
  ; FAST: store i8 [[EM]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  %e2 = extractvalue {i1, i5} %s, 0
  ret i1 %e2
}

define {i1, i5} @insert_struct({i1, i5} %s, i5 %e1) {
  ; FAST: @insert_struct.dfsan
  ; FAST: [[EM:%.*]] = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN:2]]
  ; FAST: [[SM:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: [[SM1:%.*]] = insertvalue { i8, i8 } [[SM]], i8 [[EM]], 1
  ; FAST: store { i8, i8 } [[SM1]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  %s1 = insertvalue {i1, i5} %s, i5 %e1, 1
  ret {i1, i5} %s1
}

define {i1, i1} @load_struct(ptr %p) {
  ; NO_COMBINE_LOAD_PTR: @load_struct.dfsan
  ; NO_COMBINE_LOAD_PTR: [[OL:%.*]] = or i8
  ; NO_COMBINE_LOAD_PTR: [[S0:%.*]] = insertvalue { i8, i8 } undef, i8 [[OL]], 0
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue { i8, i8 } [[S0]], i8 [[OL]], 1
  ; NO_COMBINE_LOAD_PTR: store { i8, i8 } [[S1]], ptr @__dfsan_retval_tls, align 2

  ; EVENT_CALLBACKS: @load_struct.dfsan
  ; EVENT_CALLBACKS: [[OL0:%.*]] = or i8
  ; EVENT_CALLBACKS: [[OL1:%.*]] = or i8 [[OL0]],
  ; EVENT_CALLBACKS: [[S0:%.*]] = insertvalue { i8, i8 } undef, i8 [[OL1]], 0
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i8 zeroext [[OL1]]

  %s = load {i1, i1}, ptr %p
  ret {i1, i1} %s
}

define void @store_struct(ptr %p, {i1, i1} %s) {
  ; FAST: @store_struct.dfsan
  ; FAST: [[S:%.*]] = load { i8, i8 }, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN:2]]
  ; FAST: [[E0:%.*]] = extractvalue { i8, i8 } [[S]], 0
  ; FAST: [[E1:%.*]] = extractvalue { i8, i8 } [[S]], 1
  ; FAST: [[E:%.*]] = or i8 [[E0]], [[E1]]
  ; FAST: [[P0:%.*]] = getelementptr i8, ptr [[P:%.*]], i32 0
  ; FAST: store i8 [[E]], ptr [[P0]], align 1
  ; FAST: [[P1:%.*]] = getelementptr i8, ptr [[P]], i32 1
  ; FAST: store i8 [[E]], ptr [[P1]], align 1

  ; EVENT_CALLBACKS: @store_struct.dfsan
  ; EVENT_CALLBACKS: [[OL:%.*]] = or i8
  ; EVENT_CALLBACKS: call void @__dfsan_store_callback(i8 zeroext [[OL]]

  ; COMBINE_STORE_PTR: @store_struct.dfsan
  ; COMBINE_STORE_PTR: [[PL:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; COMBINE_STORE_PTR: [[SL:%.*]] = load { i8, i8 }, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]
  ; COMBINE_STORE_PTR: [[SL0:%.*]] = extractvalue { i8, i8 } [[SL]], 0
  ; COMBINE_STORE_PTR: [[SL1:%.*]] = extractvalue { i8, i8 } [[SL]], 1
  ; COMBINE_STORE_PTR: [[SL01:%.*]] = or i8 [[SL0]], [[SL1]]
  ; COMBINE_STORE_PTR: [[E:%.*]] = or i8 [[SL01]], [[PL]]
  ; COMBINE_STORE_PTR: [[P0:%.*]] = getelementptr i8, ptr [[P:%.*]], i32 0
  ; COMBINE_STORE_PTR: store i8 [[E]], ptr [[P0]], align 1
  ; COMBINE_STORE_PTR: [[P1:%.*]] = getelementptr i8, ptr [[P]], i32 1
  ; COMBINE_STORE_PTR: store i8 [[E]], ptr [[P1]], align 1

  store {i1, i1} %s, ptr %p
  ret void
}

define i2 @extract_struct_of_aggregate11(%StructOfAggr %s) {
  ; FAST: @extract_struct_of_aggregate11.dfsan
  ; FAST: [[E:%.*]] = load { i8, [4 x i8], i8, { i8, i8 } }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[E11:%.*]] = extractvalue { i8, [4 x i8], i8, { i8, i8 } } [[E]], 1, 1
  ; FAST: store i8 [[E11]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  %e11 = extractvalue %StructOfAggr %s, 1, 1
  ret i2 %e11
}

define [4 x i2] @extract_struct_of_aggregate1(%StructOfAggr %s) {
  ; FAST: @extract_struct_of_aggregate1.dfsan
  ; FAST: [[E:%.*]] = load { i8, [4 x i8], i8, { i8, i8 } }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[E1:%.*]] = extractvalue { i8, [4 x i8], i8, { i8, i8 } } [[E]], 1
  ; FAST: store [4 x i8] [[E1]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  %e1 = extractvalue %StructOfAggr %s, 1
  ret [4 x i2] %e1
}

define <4 x i3> @extract_struct_of_aggregate2(%StructOfAggr %s) {
  ; FAST: @extract_struct_of_aggregate2.dfsan
  ; FAST: [[E:%.*]] = load { i8, [4 x i8], i8, { i8, i8 } }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[E2:%.*]] = extractvalue { i8, [4 x i8], i8, { i8, i8 } } [[E]], 2
  ; FAST: store i8 [[E2]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  %e2 = extractvalue %StructOfAggr %s, 2
  ret <4 x i3> %e2
}

define { i1, i1 } @extract_struct_of_aggregate3(%StructOfAggr %s) {
  ; FAST: @extract_struct_of_aggregate3.dfsan
  ; FAST: [[E:%.*]] = load { i8, [4 x i8], i8, { i8, i8 } }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[E3:%.*]] = extractvalue { i8, [4 x i8], i8, { i8, i8 } } [[E]], 3
  ; FAST: store { i8, i8 } [[E3]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  %e3 = extractvalue %StructOfAggr %s, 3
  ret { i1, i1 } %e3
}

define i1 @extract_struct_of_aggregate31(%StructOfAggr %s) {
  ; FAST: @extract_struct_of_aggregate31.dfsan
  ; FAST: [[E:%.*]] = load { i8, [4 x i8], i8, { i8, i8 } }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[E31:%.*]] = extractvalue { i8, [4 x i8], i8, { i8, i8 } } [[E]], 3, 1
  ; FAST: store i8 [[E31]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  %e31 = extractvalue %StructOfAggr %s, 3, 1
  ret i1 %e31
}

define %StructOfAggr @insert_struct_of_aggregate11(%StructOfAggr %s, i2 %e11) {
  ; FAST: @insert_struct_of_aggregate11.dfsan
  ; FAST: [[E11:%.*]]  = load i8, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 8) to ptr), align [[ALIGN:2]]
  ; FAST: [[S:%.*]] = load { i8, [4 x i8], i8, { i8, i8 } }, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: [[S1:%.*]] = insertvalue { i8, [4 x i8], i8, { i8, i8 } } [[S]], i8 [[E11]], 1, 1
  ; FAST: store { i8, [4 x i8], i8, { i8, i8 } } [[S1]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  %s1 = insertvalue %StructOfAggr %s, i2 %e11, 1, 1
  ret %StructOfAggr %s1
}

define {ptr, i32} @call_struct({ptr, i32} %s) {
  ; FAST: @call_struct.dfsan
  ; FAST: [[S:%.*]] = load { i8, i8 }, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: store { i8, i8 } [[S]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: %_dfsret = load { i8, i8 }, ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; FAST: store { i8, i8 } %_dfsret, ptr @__dfsan_retval_tls, align [[ALIGN]]

  %r = call {ptr, i32} @pass_struct({ptr, i32} %s)
  ret {ptr, i32} %r
}

declare %StructOfAggr @fun_with_many_aggr_args(<2 x i7> %v, [2 x i5] %a, {i3, i3} %s)

define %StructOfAggr @call_many_aggr_args(<2 x i7> %v, [2 x i5] %a, {i3, i3} %s) {
  ; FAST: @call_many_aggr_args.dfsan
  ; FAST: [[S:%.*]] = load { i8, i8 }, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align [[ALIGN:2]]
  ; FAST: [[A:%.*]] = load [2 x i8], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]
  ; FAST: [[V:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: store i8 [[V]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: store [2 x i8] [[A]], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 2) to ptr), align [[ALIGN]]
  ; FAST: store { i8, i8 } [[S]], ptr inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align [[ALIGN]]
  ; FAST: %_dfsret = load { i8, [4 x i8], i8, { i8, i8 } }, ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; FAST: store { i8, [4 x i8], i8, { i8, i8 } } %_dfsret, ptr @__dfsan_retval_tls, align [[ALIGN]]

  %r = call %StructOfAggr @fun_with_many_aggr_args(<2 x i7> %v, [2 x i5] %a, {i3, i3} %s)
  ret %StructOfAggr %r
}
