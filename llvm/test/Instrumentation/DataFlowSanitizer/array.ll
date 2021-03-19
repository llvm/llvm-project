; RUN: opt < %s -dfsan -S | FileCheck %s --check-prefixes=CHECK,LEGACY
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-event-callbacks=true -S | FileCheck %s --check-prefixes=CHECK,EVENT_CALLBACKS
; RUN: opt < %s -dfsan -dfsan-fast-8-labels=true -dfsan-event-callbacks=true -S | FileCheck %s --check-prefixes=CHECK,EVENT_CALLBACKS
; RUN: opt < %s -dfsan -dfsan-args-abi -S | FileCheck %s --check-prefixes=CHECK,ARGS_ABI
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefixes=CHECK,FAST
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s --check-prefixes=CHECK,NO_COMBINE_LOAD_PTR
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-store=true -S | FileCheck %s --check-prefixes=CHECK,COMBINE_STORE_PTR
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-debug-nonzero-labels -S | FileCheck %s --check-prefixes=CHECK,DEBUG_NONZERO_LABELS
; RUN: opt < %s -dfsan -dfsan-fast-8-labels=true -S | FileCheck %s --check-prefixes=CHECK,FAST
; RUN: opt < %s -dfsan -dfsan-fast-8-labels=true -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s --check-prefixes=CHECK,NO_COMBINE_LOAD_PTR
; RUN: opt < %s -dfsan -dfsan-fast-8-labels=true -dfsan-combine-pointer-labels-on-store=true -S | FileCheck %s --check-prefixes=CHECK,COMBINE_STORE_PTR
; RUN: opt < %s -dfsan -dfsan-fast-8-labels=true -dfsan-debug-nonzero-labels -S | FileCheck %s --check-prefixes=CHECK,DEBUG_NONZERO_LABELS
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
; CHECK: @__dfsan_shadow_width_bits = weak_odr constant i32 [[#SBITS:]]
; CHECK: @__dfsan_shadow_width_bytes = weak_odr constant i32 [[#SBYTES:]]

define [4 x i8] @pass_array([4 x i8] %a) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$pass_array"
  ; NO_COMBINE_LOAD_PTR: %1 = load [4 x i[[#SBITS]]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [4 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store [4 x i[[#SBITS]]] %1, [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [4 x i[[#SBITS]]]*), align [[ALIGN]]

  ; ARGS_ABI: @"dfs$pass_array"
  ; ARGS_ABI: ret { [4 x i8], i[[#SBITS]] }

  ; DEBUG_NONZERO_LABELS: @"dfs$pass_array"
  ; DEBUG_NONZERO_LABELS: [[L:%.*]] = load [4 x i[[#SBITS]]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [4 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; DEBUG_NONZERO_LABELS: [[L0:%.*]] = extractvalue [4 x i[[#SBITS]]] [[L]], 0
  ; DEBUG_NONZERO_LABELS: [[L1:%.*]] = extractvalue [4 x i[[#SBITS]]] [[L]], 1
  ; DEBUG_NONZERO_LABELS: [[L01:%.*]] = or i[[#SBITS]] [[L0]], [[L1]]
  ; DEBUG_NONZERO_LABELS: [[L2:%.*]] = extractvalue [4 x i[[#SBITS]]] [[L]], 2
  ; DEBUG_NONZERO_LABELS: [[L012:%.*]] = or i[[#SBITS]] [[L01]], [[L2]]
  ; DEBUG_NONZERO_LABELS: [[L3:%.*]] = extractvalue [4 x i[[#SBITS]]] [[L]], 3
  ; DEBUG_NONZERO_LABELS: [[L0123:%.*]] = or i[[#SBITS]] [[L012]], [[L3]]
  ; DEBUG_NONZERO_LABELS: {{.*}} = icmp ne i[[#SBITS]] [[L0123]], 0
  ; DEBUG_NONZERO_LABELS: call void @__dfsan_nonzero_label()

  ret [4 x i8] %a
}

%ArrayOfStruct = type [4 x {i8*, i32}]

define %ArrayOfStruct @pass_array_of_struct(%ArrayOfStruct %as) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$pass_array_of_struct"
  ; NO_COMBINE_LOAD_PTR: %1 = load [4 x { i[[#SBITS]], i[[#SBITS]] }], [4 x { i[[#SBITS]], i[[#SBITS]] }]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [4 x { i[[#SBITS]], i[[#SBITS]] }]*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store [4 x { i[[#SBITS]], i[[#SBITS]] }] %1, [4 x { i[[#SBITS]], i[[#SBITS]] }]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [4 x { i[[#SBITS]], i[[#SBITS]] }]*), align [[ALIGN]]

  ; ARGS_ABI: @"dfs$pass_array_of_struct"
  ; ARGS_ABI: ret { [4 x { i8*, i32 }], i[[#SBITS]] }
  ret %ArrayOfStruct %as
}

define [4 x i1]* @alloca_ret_array() {
  ; NO_COMBINE_LOAD_PTR: @"dfs$alloca_ret_array"
  ; NO_COMBINE_LOAD_PTR: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  %p = alloca [4 x i1]
  ret [4 x i1]* %p
}

define [4 x i1] @load_alloca_array() {
  ; NO_COMBINE_LOAD_PTR-LABEL: @"dfs$load_alloca_array"
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R:]] = alloca i[[#SBITS]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: %p = alloca [4 x i1]
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+1]] = load i[[#SBITS]], i[[#SBITS]]* %[[#R]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+2]] = insertvalue [4 x i[[#SBITS]]] undef, i[[#SBITS]] %[[#R+1]], 0
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+3]] = insertvalue [4 x i[[#SBITS]]] %[[#R+2]], i[[#SBITS]] %[[#R+1]], 1
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+4]] = insertvalue [4 x i[[#SBITS]]] %[[#R+3]], i[[#SBITS]] %[[#R+1]], 2
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+5]] = insertvalue [4 x i[[#SBITS]]] %[[#R+4]], i[[#SBITS]] %[[#R+1]], 3
  ; NO_COMBINE_LOAD_PTR-NEXT: %a = load [4 x i1], [4 x i1]* %p
  ; NO_COMBINE_LOAD_PTR-NEXT: store [4 x i[[#SBITS]]] %[[#R+5]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [4 x i[[#SBITS]]]*), align 2
  ; NO_COMBINE_LOAD_PTR-NEXT: ret [4 x i1] %a

  %p = alloca [4 x i1]
  %a = load [4 x i1], [4 x i1]* %p
  ret [4 x i1] %a
}

define [0 x i1] @load_array0([0 x i1]* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_array0"
  ; NO_COMBINE_LOAD_PTR: store [0 x i[[#SBITS]]] zeroinitializer, [0 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [0 x i[[#SBITS]]]*), align 2
  %a = load [0 x i1], [0 x i1]* %p
  ret [0 x i1] %a
}

define [1 x i1] @load_array1([1 x i1]* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_array1"
  ; NO_COMBINE_LOAD_PTR: [[L:%.*]] = load i[[#SBITS]],
  ; NO_COMBINE_LOAD_PTR: [[S:%.*]] = insertvalue [1 x i[[#SBITS]]] undef, i[[#SBITS]] [[L]], 0
  ; NO_COMBINE_LOAD_PTR: store [1 x i[[#SBITS]]] [[S]], [1 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [1 x i[[#SBITS]]]*), align 2

  ; EVENT_CALLBACKS: @"dfs$load_array1"
  ; EVENT_CALLBACKS: [[L:%.*]] = or i[[#SBITS]]
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i[[#SBITS]] [[L]], i8* {{.*}})

  ; FAST: @"dfs$load_array1"
  ; FAST: [[P:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; FAST: [[L:%.*]] = load i[[#SBITS]], i[[#SBITS]]* {{.*}}, align [[#SBYTES]]
  ; FAST: [[U:%.*]] = or i[[#SBITS]] [[L]], [[P]]
  ; FAST: [[S1:%.*]] = insertvalue [1 x i[[#SBITS]]] undef, i[[#SBITS]] [[U]], 0
  ; FAST: store [1 x i[[#SBITS]]] [[S1]], [1 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [1 x i[[#SBITS]]]*), align [[ALIGN]]

  ; LEGACY: @"dfs$load_array1"
  ; LEGACY: [[P:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; LEGACY: [[L:%.*]] = load i[[#SBITS]], i[[#SBITS]]* {{.*}}, align [[#SBYTES]]
  ; LEGACY: [[U:%.*]] = call zeroext i[[#SBITS]] @__dfsan_union(i[[#SBITS]] zeroext [[L]], i[[#SBITS]] zeroext [[P]])
  ; LEGACY: [[PH:%.*]] = phi i[[#SBITS]] [ [[U]], {{.*}} ], [ [[L]], {{.*}} ]
  ; LEGACY: store i[[#SBITS]] [[PH]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]

  %a = load [1 x i1], [1 x i1]* %p
  ret [1 x i1] %a
}

define [2 x i1] @load_array2([2 x i1]* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_array2"
  ; NO_COMBINE_LOAD_PTR: [[P1:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[P0:%.*]], i64 1
  ; NO_COMBINE_LOAD_PTR-DAG: [[E1:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[P1]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR-DAG: [[E0:%.*]] = load i[[#SBITS]], i[[#SBITS]]* [[P0]], align [[#SBYTES]]
  ; NO_COMBINE_LOAD_PTR: [[U:%.*]] = or i[[#SBITS]] [[E0]], [[E1]]
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue [2 x i[[#SBITS]]] undef, i[[#SBITS]] [[U]], 0
  ; NO_COMBINE_LOAD_PTR: [[S2:%.*]] = insertvalue [2 x i[[#SBITS]]] [[S1]], i[[#SBITS]] [[U]], 1
  ; NO_COMBINE_LOAD_PTR: store [2 x i[[#SBITS]]] [[S2]], [2 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]

  ; EVENT_CALLBACKS: @"dfs$load_array2"
  ; EVENT_CALLBACKS: [[O1:%.*]] = or i[[#SBITS]]
  ; EVENT_CALLBACKS: [[O2:%.*]] = or i[[#SBITS]] [[O1]]
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i[[#SBITS]] [[O2]], i8* {{.*}})

  ; FAST: @"dfs$load_array2"
  ; FAST: [[P:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; FAST: [[O:%.*]] = or i[[#SBITS]]
  ; FAST: [[U:%.*]] = or i[[#SBITS]] [[O]], [[P]]
  ; FAST: [[S:%.*]] = insertvalue [2 x i[[#SBITS]]] undef, i[[#SBITS]] [[U]], 0
  ; FAST: [[S1:%.*]] = insertvalue [2 x i[[#SBITS]]] [[S]], i[[#SBITS]] [[U]], 1
  ; FAST: store [2 x i[[#SBITS]]] [[S1]], [2 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [2 x i[[#SBITS]]]*), align [[ALIGN]]
  %a = load [2 x i1], [2 x i1]* %p
  ret [2 x i1] %a
}

define [4 x i1] @load_array4([4 x i1]* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_array4"
  ; NO_COMBINE_LOAD_PTR: [[T:%.*]] = trunc i[[#mul(4, SBITS)]] {{.*}} to i[[#SBITS]]
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue [4 x i[[#SBITS]]] undef, i[[#SBITS]] [[T]], 0
  ; NO_COMBINE_LOAD_PTR: [[S2:%.*]] = insertvalue [4 x i[[#SBITS]]] [[S1]], i[[#SBITS]] [[T]], 1
  ; NO_COMBINE_LOAD_PTR: [[S3:%.*]] = insertvalue [4 x i[[#SBITS]]] [[S2]], i[[#SBITS]] [[T]], 2
  ; NO_COMBINE_LOAD_PTR: [[S4:%.*]] = insertvalue [4 x i[[#SBITS]]] [[S3]], i[[#SBITS]] [[T]], 3
  ; NO_COMBINE_LOAD_PTR: store [4 x i[[#SBITS]]] [[S4]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [4 x i[[#SBITS]]]*), align 2

  ; EVENT_CALLBACKS: @"dfs$load_array4"
  ; EVENT_CALLBACKS: [[O0:%.*]] = or i[[#mul(4, SBITS)]]
  ; EVENT_CALLBACKS: [[O1:%.*]] = or i[[#mul(4, SBITS)]] [[O0]]
  ; EVENT_CALLBACKS: [[O2:%.*]] = trunc i[[#mul(4, SBITS)]] [[O1]] to i[[#SBITS]]
  ; EVENT_CALLBACKS: [[O3:%.*]] = or i[[#SBITS]] [[O2]]
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i[[#SBITS]] [[O3]], i8* {{.*}})

  ; FAST: @"dfs$load_array4"
  ; FAST: [[T:%.*]] = trunc i[[#mul(4, SBITS)]] {{.*}} to i[[#SBITS]]
  ; FAST: [[O:%.*]] = or i[[#SBITS]] [[T]]
  ; FAST: [[S1:%.*]] = insertvalue [4 x i[[#SBITS]]] undef, i[[#SBITS]] [[O]], 0
  ; FAST: [[S2:%.*]] = insertvalue [4 x i[[#SBITS]]] [[S1]], i[[#SBITS]] [[O]], 1
  ; FAST: [[S3:%.*]] = insertvalue [4 x i[[#SBITS]]] [[S2]], i[[#SBITS]] [[O]], 2
  ; FAST: [[S4:%.*]] = insertvalue [4 x i[[#SBITS]]] [[S3]], i[[#SBITS]] [[O]], 3
  ; FAST: store [4 x i[[#SBITS]]] [[S4]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [4 x i[[#SBITS]]]*), align 2

  ; LEGACY: @"dfs$load_array4"
  ; LEGACY: [[P:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; LEGACY: [[PH1:%.*]] = phi i[[#SBITS]]
  ; LEGACY: [[U:%.*]] = call zeroext i[[#SBITS]] @__dfsan_union(i[[#SBITS]] zeroext [[PH1]], i[[#SBITS]] zeroext [[P]])
  ; LEGACY: [[PH:%.*]] = phi i[[#SBITS]] [ [[U]], {{.*}} ], [ [[PH1]], {{.*}} ]
  ; LEGACY: store i[[#SBITS]] [[PH]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align [[ALIGN]]

  %a = load [4 x i1], [4 x i1]* %p
  ret [4 x i1] %a
}

define i1 @extract_array([4 x i1] %a) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$extract_array"
  ; NO_COMBINE_LOAD_PTR: [[AM:%.*]] = load [4 x i[[#SBITS]]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [4 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: [[EM:%.*]] = extractvalue [4 x i[[#SBITS]]] [[AM]], 2
  ; NO_COMBINE_LOAD_PTR: store i[[#SBITS]] [[EM]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  %e2 = extractvalue [4 x i1] %a, 2
  ret i1 %e2
}

define [4 x i1] @insert_array([4 x i1] %a, i1 %e2) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$insert_array"
  ; NO_COMBINE_LOAD_PTR: [[EM:%.*]] = load i[[#SBITS]], i[[#SBITS]]*
  ; NO_COMBINE_LOAD_PTR-SAME: inttoptr (i64 add (i64 ptrtoint ([[TLS_ARR]]* @__dfsan_arg_tls to i64), i64 [[#mul(4, SBYTES)]]) to i[[#SBITS]]*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: [[AM:%.*]] = load [4 x i[[#SBITS]]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [4 x i[[#SBITS]]]*), align [[ALIGN]]
  ; NO_COMBINE_LOAD_PTR: [[AM1:%.*]] = insertvalue [4 x i[[#SBITS]]] [[AM]], i[[#SBITS]] [[EM]], 0
  ; NO_COMBINE_LOAD_PTR: store [4 x i[[#SBITS]]] [[AM1]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [4 x i[[#SBITS]]]*), align [[ALIGN]]
  %a1 = insertvalue [4 x i1] %a, i1 %e2, 0
  ret [4 x i1] %a1
}

define void @store_alloca_array([4 x i1] %a) {
  ; FAST: @"dfs$store_alloca_array"
  ; FAST: [[S:%.*]] = load [4 x i[[#SBITS]]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [4 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; FAST: [[SP:%.*]] = alloca i[[#SBITS]], align [[#SBYTES]]
  ; FAST: [[E0:%.*]] = extractvalue [4 x i[[#SBITS]]] [[S]], 0
  ; FAST: [[E1:%.*]] = extractvalue [4 x i[[#SBITS]]] [[S]], 1
  ; FAST: [[E01:%.*]] = or i[[#SBITS]] [[E0]], [[E1]]
  ; FAST: [[E2:%.*]] = extractvalue [4 x i[[#SBITS]]] [[S]], 2
  ; FAST: [[E012:%.*]] = or i[[#SBITS]] [[E01]], [[E2]]
  ; FAST: [[E3:%.*]] = extractvalue [4 x i[[#SBITS]]] [[S]], 3
  ; FAST: [[E0123:%.*]] = or i[[#SBITS]] [[E012]], [[E3]]
  ; FAST: store i[[#SBITS]] [[E0123]], i[[#SBITS]]* [[SP]], align [[#SBYTES]]
  %p = alloca [4 x i1]
  store [4 x i1] %a, [4 x i1]* %p
  ret void
}

define void @store_zero_array([4 x i1]* %p) {
  ; FAST: @"dfs$store_zero_array"
  ; FAST: store i[[#mul(4, SBITS)]] 0, i[[#mul(4, SBITS)]]* {{.*}}
  store [4 x i1] zeroinitializer, [4 x i1]* %p
  ret void
}

define void @store_array2([2 x i1] %a, [2 x i1]* %p) {
  ; LEGACY: @"dfs$store_array2"
  ; LEGACY: [[S:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; LEGACY: [[SP0:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[SP:%.*]], i32 0
  ; LEGACY: store i[[#SBITS]] [[S]], i[[#SBITS]]* [[SP0]], align [[#SBYTES]]
  ; LEGACY: [[SP1:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[SP]], i32 1
  ; LEGACY: store i[[#SBITS]] [[S]], i[[#SBITS]]* [[SP1]], align [[#SBYTES]]

  ; EVENT_CALLBACKS: @"dfs$store_array2"
  ; EVENT_CALLBACKS: [[E12:%.*]] = or i[[#SBITS]]
  ; EVENT_CALLBACKS: [[P:%.*]] = bitcast [2 x i1]* %p to i8*
  ; EVENT_CALLBACKS: call void @__dfsan_store_callback(i[[#SBITS]] [[E12]], i8* [[P]])

  ; FAST: @"dfs$store_array2"
  ; FAST: [[S:%.*]] = load [2 x i[[#SBITS]]], [2 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [2 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; FAST: [[E1:%.*]] = extractvalue [2 x i[[#SBITS]]] [[S]], 0
  ; FAST: [[E2:%.*]] = extractvalue [2 x i[[#SBITS]]] [[S]], 1
  ; FAST: [[E12:%.*]] = or i[[#SBITS]] [[E1]], [[E2]]
  ; FAST: [[SP0:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[SP:%.*]], i32 0
  ; FAST: store i[[#SBITS]] [[E12]], i[[#SBITS]]* [[SP0]], align [[#SBYTES]]
  ; FAST: [[SP1:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[SP]], i32 1
  ; FAST: store i[[#SBITS]] [[E12]], i[[#SBITS]]* [[SP1]], align [[#SBYTES]]

  ; COMBINE_STORE_PTR: @"dfs$store_array2"
  ; COMBINE_STORE_PTR: [[O:%.*]] = or i[[#SBITS]]
  ; COMBINE_STORE_PTR: [[U:%.*]] = or i[[#SBITS]] [[O]]
  ; COMBINE_STORE_PTR: [[P1:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[P:%.*]], i32 0
  ; COMBINE_STORE_PTR: store i[[#SBITS]] [[U]], i[[#SBITS]]* [[P1]], align [[#SBYTES]]
  ; COMBINE_STORE_PTR: [[P2:%.*]] = getelementptr i[[#SBITS]], i[[#SBITS]]* [[P]], i32 1
  ; COMBINE_STORE_PTR: store i[[#SBITS]] [[U]], i[[#SBITS]]* [[P2]], align [[#SBYTES]]

  store [2 x i1] %a, [2 x i1]* %p
  ret void
}

define void @store_array17([17 x i1] %a, [17 x i1]* %p) {
  ; FAST: @"dfs$store_array17"
  ; FAST: %[[#R:]]   = load [17 x i[[#SBITS]]], [17 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [17 x i[[#SBITS]]]*), align 2
  ; FAST: %[[#R+1]]  = extractvalue [17 x i[[#SBITS]]] %[[#R]], 0
  ; FAST: %[[#R+2]]  = extractvalue [17 x i[[#SBITS]]] %[[#R]], 1
  ; FAST: %[[#R+3]]  = or i[[#SBITS]] %[[#R+1]], %[[#R+2]]
  ; FAST: %[[#R+4]]  = extractvalue [17 x i[[#SBITS]]] %[[#R]], 2
  ; FAST: %[[#R+5]]  = or i[[#SBITS]] %[[#R+3]], %[[#R+4]]
  ; FAST: %[[#R+6]]  = extractvalue [17 x i[[#SBITS]]] %[[#R]], 3
  ; FAST: %[[#R+7]]  = or i[[#SBITS]] %[[#R+5]], %[[#R+6]]
  ; FAST: %[[#R+8]]  = extractvalue [17 x i[[#SBITS]]] %[[#R]], 4
  ; FAST: %[[#R+9]]  = or i[[#SBITS]] %[[#R+7]], %[[#R+8]]
  ; FAST: %[[#R+10]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 5
  ; FAST: %[[#R+11]] = or i[[#SBITS]] %[[#R+9]], %[[#R+10]]
  ; FAST: %[[#R+12]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 6
  ; FAST: %[[#R+13]] = or i[[#SBITS]] %[[#R+11]], %[[#R+12]]
  ; FAST: %[[#R+14]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 7
  ; FAST: %[[#R+15]] = or i[[#SBITS]] %[[#R+13]], %[[#R+14]]
  ; FAST: %[[#R+16]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 8
  ; FAST: %[[#R+17]] = or i[[#SBITS]] %[[#R+15]], %[[#R+16]]
  ; FAST: %[[#R+18]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 9
  ; FAST: %[[#R+19]] = or i[[#SBITS]] %[[#R+17]], %[[#R+18]]
  ; FAST: %[[#R+20]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 10
  ; FAST: %[[#R+21]] = or i[[#SBITS]] %[[#R+19]], %[[#R+20]]
  ; FAST: %[[#R+22]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 11
  ; FAST: %[[#R+23]] = or i[[#SBITS]] %[[#R+21]], %[[#R+22]]
  ; FAST: %[[#R+24]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 12
  ; FAST: %[[#R+25]] = or i[[#SBITS]] %[[#R+23]], %[[#R+24]]
  ; FAST: %[[#R+26]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 13
  ; FAST: %[[#R+27]] = or i[[#SBITS]] %[[#R+25]], %[[#R+26]]
  ; FAST: %[[#R+28]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 14
  ; FAST: %[[#R+29]] = or i[[#SBITS]] %[[#R+27]], %[[#R+28]]
  ; FAST: %[[#R+30]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 15
  ; FAST: %[[#R+31]] = or i[[#SBITS]] %[[#R+29]], %[[#R+30]]
  ; FAST: %[[#R+32]] = extractvalue [17 x i[[#SBITS]]] %[[#R]], 16
  ; FAST: %[[#R+33]] = or i[[#SBITS]] %[[#R+31]], %[[#R+32]]
  ; FAST: %[[#VREG:]]  = insertelement <8 x i[[#SBITS]]> undef, i[[#SBITS]] %[[#R+33]], i32 0
  ; FAST: %[[#VREG+1]] = insertelement <8 x i[[#SBITS]]> %[[#VREG]], i[[#SBITS]] %[[#R+33]], i32 1
  ; FAST: %[[#VREG+2]] = insertelement <8 x i[[#SBITS]]> %[[#VREG+1]], i[[#SBITS]] %[[#R+33]], i32 2
  ; FAST: %[[#VREG+3]] = insertelement <8 x i[[#SBITS]]> %[[#VREG+2]], i[[#SBITS]] %[[#R+33]], i32 3
  ; FAST: %[[#VREG+4]] = insertelement <8 x i[[#SBITS]]> %[[#VREG+3]], i[[#SBITS]] %[[#R+33]], i32 4
  ; FAST: %[[#VREG+5]] = insertelement <8 x i[[#SBITS]]> %[[#VREG+4]], i[[#SBITS]] %[[#R+33]], i32 5
  ; FAST: %[[#VREG+6]] = insertelement <8 x i[[#SBITS]]> %[[#VREG+5]], i[[#SBITS]] %[[#R+33]], i32 6
  ; FAST: %[[#VREG+7]] = insertelement <8 x i[[#SBITS]]> %[[#VREG+6]], i[[#SBITS]] %[[#R+33]], i32 7
  ; FAST: %[[#VREG+8]] = bitcast i[[#SBITS]]* %[[P:.*]] to <8 x i[[#SBITS]]>*
  ; FAST: %[[#VREG+9]]  = getelementptr <8 x i[[#SBITS]]>, <8 x i[[#SBITS]]>* %[[#VREG+8]], i32 0
  ; FAST: store <8 x i[[#SBITS]]> %[[#VREG+7]], <8 x i[[#SBITS]]>* %[[#VREG+9]], align [[#SBYTES]]
  ; FAST: %[[#VREG+10]] = getelementptr <8 x i[[#SBITS]]>, <8 x i[[#SBITS]]>* %[[#VREG+8]], i32 1
  ; FAST: store <8 x i[[#SBITS]]> %[[#VREG+7]], <8 x i[[#SBITS]]>* %[[#VREG+10]], align [[#SBYTES]]
  ; FAST: %[[#VREG+11]] = getelementptr i[[#SBITS]], i[[#SBITS]]* %[[P]], i32 16
  ; FAST: store i[[#SBITS]] %[[#R+33]], i[[#SBITS]]* %[[#VREG+11]], align [[#SBYTES]]
  store [17 x i1] %a, [17 x i1]* %p
  ret void
}

define [2 x i32] @const_array() {
  ; FAST: @"dfs$const_array"
  ; FAST: store [2 x i[[#SBITS]]] zeroinitializer, [2 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [2 x i[[#SBITS]]]*), align 2
  ret [2 x i32] [ i32 42, i32 11 ]
}

define [4 x i8] @call_array([4 x i8] %a) {
  ; FAST-LABEL: @"dfs$call_array"
  ; FAST: %[[#R:]] = load [4 x i[[#SBITS]]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [4 x i[[#SBITS]]]*), align [[ALIGN:2]]
  ; FAST: store [4 x i[[#SBITS]]] %[[#R]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to [4 x i[[#SBITS]]]*), align [[ALIGN]]
  ; FAST: %_dfsret = load [4 x i[[#SBITS]]], [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [4 x i[[#SBITS]]]*), align [[ALIGN]]
  ; FAST: store [4 x i[[#SBITS]]] %_dfsret, [4 x i[[#SBITS]]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to [4 x i[[#SBITS]]]*), align [[ALIGN]]

  %r = call [4 x i8] @pass_array([4 x i8] %a)
  ret [4 x i8] %r
}

%LargeArr = type [1000 x i8]

define i8 @fun_with_large_args(i1 %i, %LargeArr %a) {
  ; FAST: @"dfs$fun_with_large_args"
  ; FAST: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  %r = extractvalue %LargeArr %a, 0
  ret i8 %r
}

define %LargeArr @fun_with_large_ret() {
  ; FAST: @"dfs$fun_with_large_ret"
  ; FAST-NEXT: ret  [1000 x i8] zeroinitializer
  ret %LargeArr zeroinitializer
}

define i8 @call_fun_with_large_ret() {
  ; FAST: @"dfs$call_fun_with_large_ret"
  ; FAST: store i[[#SBITS]] 0, i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_retval_tls to i[[#SBITS]]*), align 2
  %r = call %LargeArr @fun_with_large_ret()
  %e = extractvalue %LargeArr %r, 0
  ret i8 %e
}

define i8 @call_fun_with_large_args(i1 %i, %LargeArr %a) {
  ; FAST: @"dfs$call_fun_with_large_args"
  ; FAST: [[I:%.*]] = load i[[#SBITS]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN:2]]
  ; FAST: store i[[#SBITS]] [[I]], i[[#SBITS]]* bitcast ([[TLS_ARR]]* @__dfsan_arg_tls to i[[#SBITS]]*), align [[ALIGN]]
  ; FAST: %r = call i8 @"dfs$fun_with_large_args"(i1 %i, [1000 x i8] %a)

  %r = call i8 @fun_with_large_args(i1 %i, %LargeArr %a)
  ret i8 %r
}
