; RUN: opt < %s -passes=dfsan -dfsan-event-callbacks=true -S | FileCheck %s --check-prefixes=CHECK,EVENT_CALLBACKS
; RUN: opt < %s -passes=dfsan -S | FileCheck %s --check-prefixes=CHECK,FAST
; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-load=false -S | FileCheck %s --check-prefixes=CHECK,NO_COMBINE_LOAD_PTR
; RUN: opt < %s -passes=dfsan -dfsan-combine-pointer-labels-on-store=true -S | FileCheck %s --check-prefixes=CHECK,COMBINE_STORE_PTR
; RUN: opt < %s -passes=dfsan -dfsan-debug-nonzero-labels -S | FileCheck %s --check-prefixes=CHECK,DEBUG_NONZERO_LABELS
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @__dfsan_arg_tls = external thread_local(initialexec) global [[TLS_ARR:\[100 x i64\]]]
; CHECK: @__dfsan_retval_tls = external thread_local(initialexec) global [[TLS_ARR]]
define [4 x i8] @pass_array([4 x i8] %a) {
  ; NO_COMBINE_LOAD_PTR: @pass_array.dfsan
  ; NO_COMBINE_LOAD_PTR: %1 = load [4 x i8], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store [4 x i8] %1, ptr @__dfsan_retval_tls, align [[ALIGN]]

  ; DEBUG_NONZERO_LABELS: @pass_array.dfsan
  ; DEBUG_NONZERO_LABELS: [[L:%.*]] = load [4 x i8], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; DEBUG_NONZERO_LABELS: [[L0:%.*]] = extractvalue [4 x i8] [[L]], 0
  ; DEBUG_NONZERO_LABELS: [[L1:%.*]] = extractvalue [4 x i8] [[L]], 1
  ; DEBUG_NONZERO_LABELS: [[L01:%.*]] = or i8 [[L0]], [[L1]]
  ; DEBUG_NONZERO_LABELS: [[L2:%.*]] = extractvalue [4 x i8] [[L]], 2
  ; DEBUG_NONZERO_LABELS: [[L012:%.*]] = or i8 [[L01]], [[L2]]
  ; DEBUG_NONZERO_LABELS: [[L3:%.*]] = extractvalue [4 x i8] [[L]], 3
  ; DEBUG_NONZERO_LABELS: [[L0123:%.*]] = or i8 [[L012]], [[L3]]
  ; DEBUG_NONZERO_LABELS: {{.*}} = icmp ne i8 [[L0123]], 0
  ; DEBUG_NONZERO_LABELS: call void @__dfsan_nonzero_label()

  ret [4 x i8] %a
}

%ArrayOfStruct = type [4 x {ptr, i32}]

define %ArrayOfStruct @pass_array_of_struct(%ArrayOfStruct %as) {
  ; NO_COMBINE_LOAD_PTR: @pass_array_of_struct.dfsan
  ; NO_COMBINE_LOAD_PTR: %1 = load [4 x { i8, i8 }], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: store [4 x { i8, i8 }] %1, ptr @__dfsan_retval_tls, align [[ALIGN]]

  ret %ArrayOfStruct %as
}

define ptr @alloca_ret_array() {
  ; NO_COMBINE_LOAD_PTR: @alloca_ret_array.dfsan
  ; NO_COMBINE_LOAD_PTR: store i8 0, ptr @__dfsan_retval_tls, align 2
  %p = alloca [4 x i1]
  ret ptr %p
}

define [4 x i1] @load_alloca_array() {
  ; NO_COMBINE_LOAD_PTR-LABEL: @load_alloca_array.dfsan
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R:]] = alloca i8, align 1
  ; NO_COMBINE_LOAD_PTR-NEXT: %p = alloca [4 x i1]
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+1]] = load i8, ptr %[[#R]], align 1
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+2]] = insertvalue [4 x i8] undef, i8 %[[#R+1]], 0
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+3]] = insertvalue [4 x i8] %[[#R+2]], i8 %[[#R+1]], 1
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+4]] = insertvalue [4 x i8] %[[#R+3]], i8 %[[#R+1]], 2
  ; NO_COMBINE_LOAD_PTR-NEXT: %[[#R+5]] = insertvalue [4 x i8] %[[#R+4]], i8 %[[#R+1]], 3
  ; NO_COMBINE_LOAD_PTR-NEXT: %a = load [4 x i1], ptr %p
  ; NO_COMBINE_LOAD_PTR-NEXT: store [4 x i8] %[[#R+5]], ptr @__dfsan_retval_tls, align 2
  ; NO_COMBINE_LOAD_PTR-NEXT: ret [4 x i1] %a

  %p = alloca [4 x i1]
  %a = load [4 x i1], ptr %p
  ret [4 x i1] %a
}

define [0 x i1] @load_array0(ptr %p) {
  ; NO_COMBINE_LOAD_PTR: @load_array0.dfsan
  ; NO_COMBINE_LOAD_PTR: store [0 x i8] zeroinitializer, ptr @__dfsan_retval_tls, align 2
  %a = load [0 x i1], ptr %p
  ret [0 x i1] %a
}

define [1 x i1] @load_array1(ptr %p) {
  ; NO_COMBINE_LOAD_PTR: @load_array1.dfsan
  ; NO_COMBINE_LOAD_PTR: [[L:%.*]] = load i8,
  ; NO_COMBINE_LOAD_PTR: [[S:%.*]] = insertvalue [1 x i8] undef, i8 [[L]], 0
  ; NO_COMBINE_LOAD_PTR: store [1 x i8] [[S]], ptr @__dfsan_retval_tls, align 2

  ; EVENT_CALLBACKS: @load_array1.dfsan
  ; EVENT_CALLBACKS: [[L:%.*]] = or i8
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i8 zeroext [[L]], ptr {{.*}})

  ; FAST: @load_array1.dfsan
  ; FAST: [[P:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[L:%.*]] = load i8, ptr {{.*}}, align 1
  ; FAST: [[U:%.*]] = or i8 [[L]], [[P]]
  ; FAST: [[S1:%.*]] = insertvalue [1 x i8] undef, i8 [[U]], 0
  ; FAST: store [1 x i8] [[S1]], ptr @__dfsan_retval_tls, align [[ALIGN]]

  %a = load [1 x i1], ptr %p
  ret [1 x i1] %a
}

define [2 x i1] @load_array2(ptr %p) {
  ; NO_COMBINE_LOAD_PTR: @load_array2.dfsan
  ; NO_COMBINE_LOAD_PTR: [[P1:%.*]] = getelementptr i8, ptr [[P0:%.*]], i64 1
  ; NO_COMBINE_LOAD_PTR-DAG: [[E1:%.*]] = load i8, ptr [[P1]], align 1
  ; NO_COMBINE_LOAD_PTR-DAG: [[E0:%.*]] = load i8, ptr [[P0]], align 1
  ; NO_COMBINE_LOAD_PTR: [[U:%.*]] = or i8 [[E0]], [[E1]]
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue [2 x i8] undef, i8 [[U]], 0
  ; NO_COMBINE_LOAD_PTR: [[S2:%.*]] = insertvalue [2 x i8] [[S1]], i8 [[U]], 1
  ; NO_COMBINE_LOAD_PTR: store [2 x i8] [[S2]], ptr @__dfsan_retval_tls, align [[ALIGN:2]]

  ; EVENT_CALLBACKS: @load_array2.dfsan
  ; EVENT_CALLBACKS: [[O1:%.*]] = or i8
  ; EVENT_CALLBACKS: [[O2:%.*]] = or i8 [[O1]]
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i8 zeroext [[O2]], ptr {{.*}})

  ; FAST: @load_array2.dfsan
  ; FAST: [[P:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[O:%.*]] = or i8
  ; FAST: [[U:%.*]] = or i8 [[O]], [[P]]
  ; FAST: [[S:%.*]] = insertvalue [2 x i8] undef, i8 [[U]], 0
  ; FAST: [[S1:%.*]] = insertvalue [2 x i8] [[S]], i8 [[U]], 1
  ; FAST: store [2 x i8] [[S1]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  %a = load [2 x i1], ptr %p
  ret [2 x i1] %a
}

define [4 x i1] @load_array4(ptr %p) {
  ; NO_COMBINE_LOAD_PTR: @load_array4.dfsan
  ; NO_COMBINE_LOAD_PTR: [[T:%.*]] = trunc i32 {{.*}} to i8
  ; NO_COMBINE_LOAD_PTR: [[S1:%.*]] = insertvalue [4 x i8] undef, i8 [[T]], 0
  ; NO_COMBINE_LOAD_PTR: [[S2:%.*]] = insertvalue [4 x i8] [[S1]], i8 [[T]], 1
  ; NO_COMBINE_LOAD_PTR: [[S3:%.*]] = insertvalue [4 x i8] [[S2]], i8 [[T]], 2
  ; NO_COMBINE_LOAD_PTR: [[S4:%.*]] = insertvalue [4 x i8] [[S3]], i8 [[T]], 3
  ; NO_COMBINE_LOAD_PTR: store [4 x i8] [[S4]], ptr @__dfsan_retval_tls, align 2

  ; EVENT_CALLBACKS: @load_array4.dfsan
  ; EVENT_CALLBACKS: [[O0:%.*]] = or i32
  ; EVENT_CALLBACKS: [[O1:%.*]] = or i32 [[O0]]
  ; EVENT_CALLBACKS: [[O2:%.*]] = trunc i32 [[O1]] to i8
  ; EVENT_CALLBACKS: [[O3:%.*]] = or i8 [[O2]]
  ; EVENT_CALLBACKS: call void @__dfsan_load_callback(i8 zeroext [[O3]], ptr {{.*}})

  ; FAST: @load_array4.dfsan
  ; FAST: [[T:%.*]] = trunc i32 {{.*}} to i8
  ; FAST: [[O:%.*]] = or i8 [[T]]
  ; FAST: [[S1:%.*]] = insertvalue [4 x i8] undef, i8 [[O]], 0
  ; FAST: [[S2:%.*]] = insertvalue [4 x i8] [[S1]], i8 [[O]], 1
  ; FAST: [[S3:%.*]] = insertvalue [4 x i8] [[S2]], i8 [[O]], 2
  ; FAST: [[S4:%.*]] = insertvalue [4 x i8] [[S3]], i8 [[O]], 3
  ; FAST: store [4 x i8] [[S4]], ptr @__dfsan_retval_tls, align 2

  %a = load [4 x i1], ptr %p
  ret [4 x i1] %a
}

define i1 @extract_array([4 x i1] %a) {
  ; NO_COMBINE_LOAD_PTR: @extract_array.dfsan
  ; NO_COMBINE_LOAD_PTR: [[AM:%.*]] = load [4 x i8], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: [[EM:%.*]] = extractvalue [4 x i8] [[AM]], 2
  ; NO_COMBINE_LOAD_PTR: store i8 [[EM]], ptr @__dfsan_retval_tls, align 2
  %e2 = extractvalue [4 x i1] %a, 2
  ret i1 %e2
}

define [4 x i1] @insert_array([4 x i1] %a, i1 %e2) {
  ; NO_COMBINE_LOAD_PTR: @insert_array.dfsan
  ; NO_COMBINE_LOAD_PTR: [[EM:%.*]] = load i8, ptr
  ; NO_COMBINE_LOAD_PTR-SAME: inttoptr (i64 add (i64 ptrtoint (ptr @__dfsan_arg_tls to i64), i64 4) to ptr), align [[ALIGN:2]]
  ; NO_COMBINE_LOAD_PTR: [[AM:%.*]] = load [4 x i8], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; NO_COMBINE_LOAD_PTR: [[AM1:%.*]] = insertvalue [4 x i8] [[AM]], i8 [[EM]], 0
  ; NO_COMBINE_LOAD_PTR: store [4 x i8] [[AM1]], ptr @__dfsan_retval_tls, align [[ALIGN]]
  %a1 = insertvalue [4 x i1] %a, i1 %e2, 0
  ret [4 x i1] %a1
}

define void @store_alloca_array([4 x i1] %a) {
  ; FAST: @store_alloca_array.dfsan
  ; FAST: [[S:%.*]] = load [4 x i8], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[SP:%.*]] = alloca i8, align 1
  ; FAST: [[E0:%.*]] = extractvalue [4 x i8] [[S]], 0
  ; FAST: [[E1:%.*]] = extractvalue [4 x i8] [[S]], 1
  ; FAST: [[E01:%.*]] = or i8 [[E0]], [[E1]]
  ; FAST: [[E2:%.*]] = extractvalue [4 x i8] [[S]], 2
  ; FAST: [[E012:%.*]] = or i8 [[E01]], [[E2]]
  ; FAST: [[E3:%.*]] = extractvalue [4 x i8] [[S]], 3
  ; FAST: [[E0123:%.*]] = or i8 [[E012]], [[E3]]
  ; FAST: store i8 [[E0123]], ptr [[SP]], align 1
  %p = alloca [4 x i1]
  store [4 x i1] %a, ptr %p
  ret void
}

define void @store_zero_array(ptr %p) {
  ; FAST: @store_zero_array.dfsan
  ; FAST: store i32 0, ptr {{.*}}
  store [4 x i1] zeroinitializer, ptr %p
  ret void
}

define void @store_array2([2 x i1] %a, ptr %p) {
  ; EVENT_CALLBACKS: @store_array2.dfsan
  ; EVENT_CALLBACKS: [[E12:%.*]] = or i8
  ; EVENT_CALLBACKS: call void @__dfsan_store_callback(i8 zeroext [[E12]], ptr %p)

  ; FAST: @store_array2.dfsan
  ; FAST: [[S:%.*]] = load [2 x i8], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: [[E1:%.*]] = extractvalue [2 x i8] [[S]], 0
  ; FAST: [[E2:%.*]] = extractvalue [2 x i8] [[S]], 1
  ; FAST: [[E12:%.*]] = or i8 [[E1]], [[E2]]
  ; FAST: [[SP0:%.*]] = getelementptr i8, ptr [[SP:%.*]], i32 0
  ; FAST: store i8 [[E12]], ptr [[SP0]], align 1
  ; FAST: [[SP1:%.*]] = getelementptr i8, ptr [[SP]], i32 1
  ; FAST: store i8 [[E12]], ptr [[SP1]], align 1

  ; COMBINE_STORE_PTR: @store_array2.dfsan
  ; COMBINE_STORE_PTR: [[O:%.*]] = or i8
  ; COMBINE_STORE_PTR: [[U:%.*]] = or i8 [[O]]
  ; COMBINE_STORE_PTR: [[P1:%.*]] = getelementptr i8, ptr [[P:%.*]], i32 0
  ; COMBINE_STORE_PTR: store i8 [[U]], ptr [[P1]], align 1
  ; COMBINE_STORE_PTR: [[P2:%.*]] = getelementptr i8, ptr [[P]], i32 1
  ; COMBINE_STORE_PTR: store i8 [[U]], ptr [[P2]], align 1

  store [2 x i1] %a, ptr %p
  ret void
}

define void @store_array17([17 x i1] %a, ptr %p) {
  ; FAST: @store_array17.dfsan
  ; FAST: %[[#R:]]   = load [17 x i8], ptr @__dfsan_arg_tls, align 2
  ; FAST: %[[#R+1]]  = extractvalue [17 x i8] %[[#R]], 0
  ; FAST: %[[#R+2]]  = extractvalue [17 x i8] %[[#R]], 1
  ; FAST: %[[#R+3]]  = or i8 %[[#R+1]], %[[#R+2]]
  ; FAST: %[[#R+4]]  = extractvalue [17 x i8] %[[#R]], 2
  ; FAST: %[[#R+5]]  = or i8 %[[#R+3]], %[[#R+4]]
  ; FAST: %[[#R+6]]  = extractvalue [17 x i8] %[[#R]], 3
  ; FAST: %[[#R+7]]  = or i8 %[[#R+5]], %[[#R+6]]
  ; FAST: %[[#R+8]]  = extractvalue [17 x i8] %[[#R]], 4
  ; FAST: %[[#R+9]]  = or i8 %[[#R+7]], %[[#R+8]]
  ; FAST: %[[#R+10]] = extractvalue [17 x i8] %[[#R]], 5
  ; FAST: %[[#R+11]] = or i8 %[[#R+9]], %[[#R+10]]
  ; FAST: %[[#R+12]] = extractvalue [17 x i8] %[[#R]], 6
  ; FAST: %[[#R+13]] = or i8 %[[#R+11]], %[[#R+12]]
  ; FAST: %[[#R+14]] = extractvalue [17 x i8] %[[#R]], 7
  ; FAST: %[[#R+15]] = or i8 %[[#R+13]], %[[#R+14]]
  ; FAST: %[[#R+16]] = extractvalue [17 x i8] %[[#R]], 8
  ; FAST: %[[#R+17]] = or i8 %[[#R+15]], %[[#R+16]]
  ; FAST: %[[#R+18]] = extractvalue [17 x i8] %[[#R]], 9
  ; FAST: %[[#R+19]] = or i8 %[[#R+17]], %[[#R+18]]
  ; FAST: %[[#R+20]] = extractvalue [17 x i8] %[[#R]], 10
  ; FAST: %[[#R+21]] = or i8 %[[#R+19]], %[[#R+20]]
  ; FAST: %[[#R+22]] = extractvalue [17 x i8] %[[#R]], 11
  ; FAST: %[[#R+23]] = or i8 %[[#R+21]], %[[#R+22]]
  ; FAST: %[[#R+24]] = extractvalue [17 x i8] %[[#R]], 12
  ; FAST: %[[#R+25]] = or i8 %[[#R+23]], %[[#R+24]]
  ; FAST: %[[#R+26]] = extractvalue [17 x i8] %[[#R]], 13
  ; FAST: %[[#R+27]] = or i8 %[[#R+25]], %[[#R+26]]
  ; FAST: %[[#R+28]] = extractvalue [17 x i8] %[[#R]], 14
  ; FAST: %[[#R+29]] = or i8 %[[#R+27]], %[[#R+28]]
  ; FAST: %[[#R+30]] = extractvalue [17 x i8] %[[#R]], 15
  ; FAST: %[[#R+31]] = or i8 %[[#R+29]], %[[#R+30]]
  ; FAST: %[[#R+32]] = extractvalue [17 x i8] %[[#R]], 16
  ; FAST: %[[#R+33]] = or i8 %[[#R+31]], %[[#R+32]]
  ; FAST: %[[#VREG:]]  = insertelement <8 x i8> poison, i8 %[[#R+33]], i32 0
  ; FAST: %[[#VREG+1]] = insertelement <8 x i8> %[[#VREG]], i8 %[[#R+33]], i32 1
  ; FAST: %[[#VREG+2]] = insertelement <8 x i8> %[[#VREG+1]], i8 %[[#R+33]], i32 2
  ; FAST: %[[#VREG+3]] = insertelement <8 x i8> %[[#VREG+2]], i8 %[[#R+33]], i32 3
  ; FAST: %[[#VREG+4]] = insertelement <8 x i8> %[[#VREG+3]], i8 %[[#R+33]], i32 4
  ; FAST: %[[#VREG+5]] = insertelement <8 x i8> %[[#VREG+4]], i8 %[[#R+33]], i32 5
  ; FAST: %[[#VREG+6]] = insertelement <8 x i8> %[[#VREG+5]], i8 %[[#R+33]], i32 6
  ; FAST: %[[#VREG+7]] = insertelement <8 x i8> %[[#VREG+6]], i8 %[[#R+33]], i32 7
  ; FAST: %[[#VREG+8]] = getelementptr <8 x i8>, ptr %[[P:.*]], i32 0
  ; FAST: store <8 x i8> %[[#VREG+7]], ptr %[[#VREG+8]], align 1
  ; FAST: %[[#VREG+9]] = getelementptr <8 x i8>, ptr %[[P]], i32 1
  ; FAST: store <8 x i8> %[[#VREG+7]], ptr %[[#VREG+9]], align 1
  ; FAST: %[[#VREG+10]] = getelementptr i8, ptr %[[P]], i32 16
  ; FAST: store i8 %[[#R+33]], ptr %[[#VREG+10]], align 1
  store [17 x i1] %a, ptr %p
  ret void
}

define [2 x i32] @const_array() {
  ; FAST: @const_array.dfsan
  ; FAST: store [2 x i8] zeroinitializer, ptr @__dfsan_retval_tls, align 2
  ret [2 x i32] [ i32 42, i32 11 ]
}

define [4 x i8] @call_array([4 x i8] %a) {
  ; FAST-LABEL: @call_array.dfsan
  ; FAST: %[[#R:]] = load [4 x i8], ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: store [4 x i8] %[[#R]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: %_dfsret = load [4 x i8], ptr @__dfsan_retval_tls, align [[ALIGN]]
  ; FAST: store [4 x i8] %_dfsret, ptr @__dfsan_retval_tls, align [[ALIGN]]

  %r = call [4 x i8] @pass_array([4 x i8] %a)
  ret [4 x i8] %r
}

%LargeArr = type [1000 x i8]

define i8 @fun_with_large_args(i1 %i, %LargeArr %a) {
  ; FAST: @fun_with_large_args.dfsan
  ; FAST: store i8 0, ptr @__dfsan_retval_tls, align 2
  %r = extractvalue %LargeArr %a, 0
  ret i8 %r
}

define %LargeArr @fun_with_large_ret() {
  ; FAST: @fun_with_large_ret.dfsan
  ; FAST-NEXT: ret  [1000 x i8] zeroinitializer
  ret %LargeArr zeroinitializer
}

define i8 @call_fun_with_large_ret() {
  ; FAST: @call_fun_with_large_ret.dfsan
  ; FAST: store i8 0, ptr @__dfsan_retval_tls, align 2
  %r = call %LargeArr @fun_with_large_ret()
  %e = extractvalue %LargeArr %r, 0
  ret i8 %e
}

define i8 @call_fun_with_large_args(i1 %i, %LargeArr %a) {
  ; FAST: @call_fun_with_large_args.dfsan
  ; FAST: [[I:%.*]] = load i8, ptr @__dfsan_arg_tls, align [[ALIGN:2]]
  ; FAST: store i8 [[I]], ptr @__dfsan_arg_tls, align [[ALIGN]]
  ; FAST: %r = call i8 @fun_with_large_args.dfsan(i1 %i, [1000 x i8] %a)

  %r = call i8 @fun_with_large_args(i1 %i, %LargeArr %a)
  ret i8 %r
}
