; RUN: opt < %s -dfsan -S | FileCheck %s --check-prefix=LEGACY
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-event-callbacks=true -S | FileCheck %s --check-prefix=EVENT_CALLBACKS
; RUN: opt < %s -dfsan -dfsan-args-abi -S | FileCheck %s --check-prefix=ARGS_ABI
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefix=FAST16
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s --check-prefix=NO_COMBINE_LOAD_PTR
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-combine-pointer-labels-on-store=true -S | FileCheck %s --check-prefix=COMBINE_STORE_PTR
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -dfsan-debug-nonzero-labels -S | FileCheck %s --check-prefix=DEBUG_NONZERO_LABELS
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define [4 x i8] @pass_array([4 x i8] %a) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$pass_array"
  ; NO_COMBINE_LOAD_PTR: %1 = load [4 x i16], [4 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [4 x i16]*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store [4 x i16] %1, [4 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x i16]*), align [[ALIGN]]

  ; ARGS_ABI: @"dfs$pass_array"
  ; ARGS_ABI: ret { [4 x i8], i16 }
  
  ; DEBUG_NONZERO_LABELS: @"dfs$pass_array"
  ; DEBUG_NONZERO_LABELS: [[L:%.*]] = load [4 x i16], [4 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [4 x i16]*), align [[ALIGN:2]]
  ; DEBUG_NONZERO_LABELS: [[L0:%.*]] = extractvalue [4 x i16] [[L]], 0
  ; DEBUG_NONZERO_LABELS: [[L1:%.*]] = extractvalue [4 x i16] [[L]], 1
  ; DEBUG_NONZERO_LABELS: [[L01:%.*]] = or i16 [[L0]], [[L1]]
  ; DEBUG_NONZERO_LABELS: [[L2:%.*]] = extractvalue [4 x i16] [[L]], 2
  ; DEBUG_NONZERO_LABELS: [[L012:%.*]] = or i16 [[L01]], [[L2]]
  ; DEBUG_NONZERO_LABELS: [[L3:%.*]] = extractvalue [4 x i16] [[L]], 3
  ; DEBUG_NONZERO_LABELS: [[L0123:%.*]] = or i16 [[L012]], [[L3]]
  ; DEBUG_NONZERO_LABELS: {{.*}} = icmp ne i16 [[L0123]], 0
  ; DEBUG_NONZERO_LABELS: call void @__dfsan_nonzero_label()
  
  ret [4 x i8] %a
}

%ArrayOfStruct = type [4 x {i8*, i32}]

define %ArrayOfStruct @pass_array_of_struct(%ArrayOfStruct %as) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$pass_array_of_struct"
  ; NO_COMBINE_LOAD_PTR: %1 = load [4 x { i16, i16 }], [4 x { i16, i16 }]* bitcast ([100 x i64]* @__dfsan_arg_tls to [4 x { i16, i16 }]*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store [4 x { i16, i16 }] %1, [4 x { i16, i16 }]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x { i16, i16 }]*), align [[ALIGN]]

  ; ARGS_ABI: @"dfs$pass_array_of_struct"
  ; ARGS_ABI: ret { [4 x { i8*, i32 }], i16 }
  ret %ArrayOfStruct %as
}

define [4 x i1]* @alloca_ret_array() {
  ; NO_COMBINE_LOAD_PTR: @"dfs$alloca_ret_array"
  ; NO_COMBINE_LOAD_PTR: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
  %p = alloca [4 x i1]
  ret [4 x i1]* %p
}

define [4 x i1] @load_alloca_array() {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_alloca_array"
  ; NO_COMBINE_LOAD_PTR: [[A:%.*]] = alloca i16, align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: [[M:%.*]] = load i16, i16* [[A]], align [[ALIGN]]
  ; NO_COMBINE_LOAD_PTR: [[S0:%.*]] = insertvalue [4 x i16] undef, i16 [[M]], 0
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue [4 x i16] [[S0]], i16 [[M]], 1
  ; NO_COMBINE_LOAD_PTR: [[S2:%.*]] = insertvalue [4 x i16] [[S1]], i16 [[M]], 2
  ; NO_COMBINE_LOAD_PTR: [[S3:%.*]] = insertvalue [4 x i16] [[S2]], i16 [[M]], 3
  ; NO_COMBINE_LOAD_PTR: store [4 x i16] [[S3]], [4 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x i16]*), align [[ALIGN]]
  %p = alloca [4 x i1]
  %a = load [4 x i1], [4 x i1]* %p
  ret [4 x i1] %a
}

define [0 x i1] @load_array0([0 x i1]* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_array0"
  ; NO_COMBINE_LOAD_PTR: store [0 x i16] zeroinitializer, [0 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [0 x i16]*), align 2
  %a = load [0 x i1], [0 x i1]* %p
  ret [0 x i1] %a
}

define [1 x i1] @load_array1([1 x i1]* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_array1"
  ; NO_COMBINE_LOAD_PTR: [[L:%.*]] = load i16,
  ; NO_COMBINE_LOAD_PTR: [[S:%.*]] = insertvalue [1 x i16] undef, i16 [[L]], 0
  ; NO_COMBINE_LOAD_PTR: store [1 x i16] [[S]], [1 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [1 x i16]*), align 2

  ; EVENT_CALLBACKS: @"dfs$load_array1"
  ; EVENT_CALLBACKS: [[L:%.*]] = or i16
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i16 [[L]], i8* {{.*}})

  ; FAST16: @"dfs$load_array1"
  ; FAST16: [[P:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; FAST16: [[L:%.*]] = load i16, i16* {{.*}}, align [[ALIGN]]
  ; FAST16: [[U:%.*]] = or i16 [[L]], [[P]]
  ; FAST16: [[S1:%.*]] = insertvalue [1 x i16] undef, i16 [[U]], 0
  ; FAST16: store [1 x i16] [[S1]], [1 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [1 x i16]*), align [[ALIGN]]
  
  ; LEGACY: @"dfs$load_array1"
  ; LEGACY: [[P:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; LEGACY: [[L:%.*]] = load i16, i16* {{.*}}, align [[ALIGN]]
  ; LEGACY: [[U:%.*]] = call zeroext i16 @__dfsan_union(i16 zeroext [[L]], i16 zeroext [[P]])
  ; LEGACY: [[PH:%.*]] = phi i16 [ [[U]], {{.*}} ], [ [[L]], {{.*}} ]
  ; LEGACY: store i16 [[PH]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]

  %a = load [1 x i1], [1 x i1]* %p
  ret [1 x i1] %a
}

define [2 x i1] @load_array2([2 x i1]* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_array2"
  ; NO_COMBINE_LOAD_PTR: [[P1:%.*]] = getelementptr i16, i16* [[P0:%.*]], i64 1
  ; NO_COMBINE_LOAD_PTR-DAG: [[E1:%.*]] = load i16, i16* [[P1]], align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR-DAG: [[E0:%.*]] = load i16, i16* [[P0]], align [[ALIGN]]
  ; NO_COMBINE_LOAD_PTR: [[U:%.*]] = or i16 [[E0]], [[E1]]
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue [2 x i16] undef, i16 [[U]], 0
  ; NO_COMBINE_LOAD_PTR: [[S2:%.*]] = insertvalue [2 x i16] [[S1]], i16 [[U]], 1
  ; NO_COMBINE_LOAD_PTR: store [2 x i16] [[S2]], [2 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [2 x i16]*), align [[ALIGN]]

  ; EVENT_CALLBACKS: @"dfs$load_array2"
  ; EVENT_CALLBACKS: [[O1:%.*]] = or i16
  ; EVENT_CALLBACKS: [[O2:%.*]] = or i16 [[O1]]
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i16 [[O2]], i8* {{.*}})

  ; FAST16: @"dfs$load_array2"
  ; FAST16: [[P:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; FAST16: [[O:%.*]] = or i16
  ; FAST16: [[U:%.*]] = or i16 [[O]], [[P]]
  ; FAST16: [[S:%.*]] = insertvalue [2 x i16] undef, i16 [[U]], 0
  ; FAST16: [[S1:%.*]] = insertvalue [2 x i16] [[S]], i16 [[U]], 1
  ; FAST16: store [2 x i16] [[S1]], [2 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [2 x i16]*), align [[ALIGN]]
  %a = load [2 x i1], [2 x i1]* %p
  ret [2 x i1] %a
}

define [4 x i1] @load_array4([4 x i1]* %p) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$load_array4"
  ; NO_COMBINE_LOAD_PTR: [[T:%.*]] = trunc i64 {{.*}} to i16
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue [4 x i16] undef, i16 [[T]], 0
  ; NO_COMBINE_LOAD_PTR: [[S2:%.*]] = insertvalue [4 x i16] [[S1]], i16 [[T]], 1
  ; NO_COMBINE_LOAD_PTR: [[S3:%.*]] = insertvalue [4 x i16] [[S2]], i16 [[T]], 2
  ; NO_COMBINE_LOAD_PTR: [[S4:%.*]] = insertvalue [4 x i16] [[S3]], i16 [[T]], 3
  ; NO_COMBINE_LOAD_PTR: store [4 x i16] [[S4]], [4 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x i16]*), align 2

  ; EVENT_CALLBACKS: @"dfs$load_array4"
  ; EVENT_CALLBACKS: [[O0:%.*]] = or i64
  ; EVENT_CALLBACKS: [[O1:%.*]] = or i64 [[O0]]
  ; EVENT_CALLBACKS: [[O2:%.*]] = trunc i64 [[O1]] to i16
  ; EVENT_CALLBACKS: [[O3:%.*]] = or i16 [[O2]]
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i16 [[O3]], i8* {{.*}})

  ; FAST16: @"dfs$load_array4"
  ; FAST16: [[T:%.*]] = trunc i64 {{.*}} to i16
  ; FAST16: [[O:%.*]] = or i16 [[T]]
  ; FAST16: [[S1:%.*]] = insertvalue [4 x i16] undef, i16 [[O]], 0
  ; FAST16: [[S2:%.*]] = insertvalue [4 x i16] [[S1]], i16 [[O]], 1
  ; FAST16: [[S3:%.*]] = insertvalue [4 x i16] [[S2]], i16 [[O]], 2
  ; FAST16: [[S4:%.*]] = insertvalue [4 x i16] [[S3]], i16 [[O]], 3
  ; FAST16: store [4 x i16] [[S4]], [4 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x i16]*), align 2

  ; LEGACY: @"dfs$load_array4"
  ; LEGACY: [[P:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; LEGACY: [[PH1:%.*]] = phi i16
  ; LEGACY: [[U:%.*]] = call zeroext i16 @__dfsan_union(i16 zeroext [[PH1]], i16 zeroext [[P]])
  ; LEGACY: [[PH:%.*]] = phi i16 [ [[U]], {{.*}} ], [ [[PH1]], {{.*}} ]
  ; LEGACY: store i16 [[PH]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]

  %a = load [4 x i1], [4 x i1]* %p
  ret [4 x i1] %a
}

define i1 @extract_array([4 x i1] %a) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$extract_array"
  ; NO_COMBINE_LOAD_PTR: [[AM:%.*]] = load [4 x i16], [4 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [4 x i16]*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: [[EM:%.*]] = extractvalue [4 x i16] [[AM]], 2
  ; NO_COMBINE_LOAD_PTR: store i16 [[EM]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
  %e2 = extractvalue [4 x i1] %a, 2
  ret i1 %e2
}

define [4 x i1] @insert_array([4 x i1] %a, i1 %e2) {
  ; NO_COMBINE_LOAD_PTR: @"dfs$insert_array"
  ; NO_COMBINE_LOAD_PTR: [[EM:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 8) to i16*), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: [[AM:%.*]] = load [4 x i16], [4 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [4 x i16]*), align [[ALIGN]]
  ; NO_COMBINE_LOAD_PTR: [[AM1:%.*]] = insertvalue [4 x i16] [[AM]], i16 [[EM]], 0
  ; NO_COMBINE_LOAD_PTR: store [4 x i16] [[AM1]], [4 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x i16]*), align [[ALIGN]]
  %a1 = insertvalue [4 x i1] %a, i1 %e2, 0
  ret [4 x i1] %a1
}

define void @store_alloca_array([4 x i1] %a) {
  ; FAST16: @"dfs$store_alloca_array"
  ; FAST16: [[S:%.*]] = load [4 x i16], [4 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [4 x i16]*), align [[ALIGN:2]]
  ; FAST16: [[SP:%.*]] = alloca i16, align [[ALIGN]]
  ; FAST16: [[E0:%.*]] = extractvalue [4 x i16] [[S]], 0
  ; FAST16: [[E1:%.*]] = extractvalue [4 x i16] [[S]], 1
  ; FAST16: [[E01:%.*]] = or i16 [[E0]], [[E1]]
  ; FAST16: [[E2:%.*]] = extractvalue [4 x i16] [[S]], 2
  ; FAST16: [[E012:%.*]] = or i16 [[E01]], [[E2]]
  ; FAST16: [[E3:%.*]] = extractvalue [4 x i16] [[S]], 3
  ; FAST16: [[E0123:%.*]] = or i16 [[E012]], [[E3]]
  ; FAST16: store i16 [[E0123]], i16* [[SP]], align [[ALIGN]]
  %p = alloca [4 x i1]
  store [4 x i1] %a, [4 x i1]* %p
  ret void
}

define void @store_zero_array([4 x i1]* %p) {
  ; FAST16: @"dfs$store_zero_array"
  ; FAST16: store i64 0, i64* {{.*}}, align 2
  store [4 x i1] zeroinitializer, [4 x i1]* %p
  ret void
}

define void @store_array2([2 x i1] %a, [2 x i1]* %p) {
  ; LEGACY: @"dfs$store_array2"
  ; LEGACY: [[S:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; LEGACY: [[SP0:%.*]] = getelementptr i16, i16* [[SP:%.*]], i32 0
  ; LEGACY: store i16 [[S]], i16* [[SP0]], align [[ALIGN]]
  ; LEGACY: [[SP1:%.*]] = getelementptr i16, i16* [[SP]], i32 1
  ; LEGACY: store i16 [[S]], i16* [[SP1]], align [[ALIGN]]
  
  ; EVENT_CALLBACKS: @"dfs$store_array2"
  ; EVENT_CALLBACKS: [[E12:%.*]] = or i16
  ; EVENT_CALLBACKS: [[P:%.*]] = bitcast [2 x i1]* %p to i8*
  ; EVENT_CALLBACKS: call void @__dfsan_store_callback(i16 [[E12]], i8* [[P]])
  
  ; FAST16: @"dfs$store_array2"
  ; FAST16: [[S:%.*]] = load [2 x i16], [2 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [2 x i16]*), align [[ALIGN:2]]
  ; FAST16: [[E1:%.*]] = extractvalue [2 x i16] [[S]], 0
  ; FAST16: [[E2:%.*]] = extractvalue [2 x i16] [[S]], 1
  ; FAST16: [[E12:%.*]] = or i16 [[E1]], [[E2]]
  ; FAST16: [[SP0:%.*]] = getelementptr i16, i16* [[SP:%.*]], i32 0
  ; FAST16: store i16 [[E12]], i16* [[SP0]], align [[ALIGN]]
  ; FAST16: [[SP1:%.*]] = getelementptr i16, i16* [[SP]], i32 1
  ; FAST16: store i16 [[E12]], i16* [[SP1]], align [[ALIGN]]

  ; COMBINE_STORE_PTR: @"dfs$store_array2"
  ; COMBINE_STORE_PTR: [[O:%.*]] = or i16
  ; COMBINE_STORE_PTR: [[U:%.*]] = or i16 [[O]]
  ; COMBINE_STORE_PTR: [[P1:%.*]] = getelementptr i16, i16* [[P:%.*]], i32 0
  ; COMBINE_STORE_PTR: store i16 [[U]], i16* [[P1]], align 2
  ; COMBINE_STORE_PTR: [[P2:%.*]] = getelementptr i16, i16* [[P]], i32 1
  ; COMBINE_STORE_PTR: store i16 [[U]], i16* [[P2]], align 2
  
  store [2 x i1] %a, [2 x i1]* %p
  ret void
}

define void @store_array17([17 x i1] %a, [17 x i1]* %p) {
  ; FAST16: @"dfs$store_array17"
  ; FAST16: [[AL:%.*]] = load [17 x i16], [17 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [17 x i16]*), align 2
  ; FAST16: [[AL0:%.*]] = extractvalue [17 x i16] [[AL]], 0
  ; FAST16: [[AL1:%.*]] = extractvalue [17 x i16] [[AL]], 1
  ; FAST16: [[AL_0_1:%.*]] = or i16 [[AL0]], [[AL1]]
  ; FAST16: [[AL2:%.*]] = extractvalue [17 x i16] [[AL]], 2
  ; FAST16: [[AL_0_2:%.*]] = or i16 [[AL_0_1]], [[AL2]]
  ; FAST16: [[AL3:%.*]] = extractvalue [17 x i16] [[AL]], 3
  ; FAST16: [[AL_0_3:%.*]] = or i16 [[AL_0_2]], [[AL3]]
  ; FAST16: [[AL4:%.*]] = extractvalue [17 x i16] [[AL]], 4
  ; FAST16: [[AL_0_4:%.*]] = or i16 [[AL_0_3]], [[AL4]]
  ; FAST16: [[AL5:%.*]] = extractvalue [17 x i16] [[AL]], 5
  ; FAST16: [[AL_0_5:%.*]] = or i16 %10, [[AL5]]
  ; FAST16: [[AL6:%.*]] = extractvalue [17 x i16] [[AL]], 6
  ; FAST16: [[AL_0_6:%.*]] = or i16 %12, [[AL6]]
  ; FAST16: [[AL7:%.*]] = extractvalue [17 x i16] [[AL]], 7
  ; FAST16: [[AL_0_7:%.*]] = or i16 %14, [[AL7]]
  ; FAST16: [[AL8:%.*]] = extractvalue [17 x i16] [[AL]], 8
  ; FAST16: [[AL_0_8:%.*]] = or i16 %16, [[AL8]]
  ; FAST16: [[AL9:%.*]] = extractvalue [17 x i16] [[AL]], 9
  ; FAST16: [[AL_0_9:%.*]] = or i16 %18, [[AL9]]
  ; FAST16: [[AL10:%.*]] = extractvalue [17 x i16] [[AL]], 10
  ; FAST16: [[AL_0_10:%.*]] = or i16 %20, [[AL10]]
  ; FAST16: [[AL11:%.*]] = extractvalue [17 x i16] [[AL]], 11
  ; FAST16: [[AL_0_11:%.*]] = or i16 %22, [[AL11]]
  ; FAST16: [[AL12:%.*]] = extractvalue [17 x i16] [[AL]], 12
  ; FAST16: [[AL_0_12:%.*]] = or i16 %24, [[AL12]]
  ; FAST16: [[AL13:%.*]] = extractvalue [17 x i16] [[AL]], 13
  ; FAST16: [[AL_0_13:%.*]] = or i16 %26, [[AL13]]
  ; FAST16: [[AL14:%.*]] = extractvalue [17 x i16] [[AL]], 14
  ; FAST16: [[AL_0_14:%.*]] = or i16 %28, [[AL14]]
  ; FAST16: [[AL15:%.*]] = extractvalue [17 x i16] [[AL]], 15
  ; FAST16: [[AL_0_15:%.*]] = or i16 %30, [[AL15]]
  ; FAST16: [[AL16:%.*]] = extractvalue [17 x i16] [[AL]], 16
  ; FAST16: [[AL_0_16:%.*]] = or i16 {{.*}}, [[AL16]]
  ; FAST16: [[V1:%.*]] = insertelement <8 x i16> undef, i16 [[AL_0_16]], i32 0
  ; FAST16: [[V2:%.*]] = insertelement <8 x i16> [[V1]], i16 [[AL_0_16]], i32 1
  ; FAST16: [[V3:%.*]] = insertelement <8 x i16> [[V2]], i16 [[AL_0_16]], i32 2
  ; FAST16: [[V4:%.*]] = insertelement <8 x i16> [[V3]], i16 [[AL_0_16]], i32 3
  ; FAST16: [[V5:%.*]] = insertelement <8 x i16> [[V4]], i16 [[AL_0_16]], i32 4
  ; FAST16: [[V6:%.*]] = insertelement <8 x i16> [[V5]], i16 [[AL_0_16]], i32 5
  ; FAST16: [[V7:%.*]] = insertelement <8 x i16> [[V6]], i16 [[AL_0_16]], i32 6
  ; FAST16: [[V8:%.*]] = insertelement <8 x i16> [[V7]], i16 [[AL_0_16]], i32 7
  ; FAST16: [[VP:%.*]] = bitcast i16* [[P:%.*]] to <8 x i16>*
  ; FAST16: [[VP1:%.*]] = getelementptr <8 x i16>, <8 x i16>* [[VP]], i32 0
  ; FAST16: store <8 x i16> [[V8]], <8 x i16>* [[VP1]], align [[ALIGN:2]]
  ; FAST16: [[VP2:%.*]] = getelementptr <8 x i16>, <8 x i16>* [[VP]], i32 1
  ; FAST16: store <8 x i16> [[V8]], <8 x i16>* [[VP2]], align [[ALIGN]]
  ; FAST16: [[P3:%.*]] = getelementptr i16, i16* [[P]], i32 16
  ; FAST16: store i16 [[AL_0_16]], i16* [[P3]], align [[ALIGN]]
  store [17 x i1] %a, [17 x i1]* %p
  ret void
}

define [2 x i32] @const_array() {
  ; FAST16: @"dfs$const_array"
  ; FAST16: store [2 x i16] zeroinitializer, [2 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [2 x i16]*), align 2
  ret [2 x i32] [ i32 42, i32 11 ]
}

define [4 x i8] @call_array([4 x i8] %a) {
  ; FAST16: @"dfs$call_array"
  ; FAST16: [[A:%.*]] = load [4 x i16], [4 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [4 x i16]*), align [[ALIGN:2]]
  ; FAST16: store [4 x i16] [[A]], [4 x i16]* bitcast ([100 x i64]* @__dfsan_arg_tls to [4 x i16]*), align [[ALIGN]]
  ; FAST16: %_dfsret = load [4 x i16], [4 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x i16]*), align [[ALIGN]]
  ; FAST16: store [4 x i16] %_dfsret, [4 x i16]* bitcast ([100 x i64]* @__dfsan_retval_tls to [4 x i16]*), align [[ALIGN]]

  %r = call [4 x i8] @pass_array([4 x i8] %a)
  ret [4 x i8] %r
}

%LargeArr = type [1000 x i8]

define i8 @fun_with_large_args(i1 %i, %LargeArr %a) {
  ; FAST16: @"dfs$fun_with_large_args"
  ; FAST16: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
  %r = extractvalue %LargeArr %a, 0
  ret i8 %r
}

define %LargeArr @fun_with_large_ret() {
  ; FAST16: @"dfs$fun_with_large_ret"
  ; FAST16-NEXT: ret  [1000 x i8] zeroinitializer
  ret %LargeArr zeroinitializer
}

define i8 @call_fun_with_large_ret() {
  ; FAST16: @"dfs$call_fun_with_large_ret"
  ; FAST16: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
  %r = call %LargeArr @fun_with_large_ret()
  %e = extractvalue %LargeArr %r, 0
  ret i8 %e
}

define i8 @call_fun_with_large_args(i1 %i, %LargeArr %a) {
  ; FAST16: @"dfs$call_fun_with_large_args"
  ; FAST16: [[I:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; FAST16: store i16 [[I]], i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; FAST16: %r = call i8 @"dfs$fun_with_large_args"(i1 %i, [1000 x i8] %a)
  
  %r = call i8 @fun_with_large_args(i1 %i, %LargeArr %a)
  ret i8 %r
}
