; RUN: opt < %s -dfsan -S | FileCheck %s --check-prefix=LEGACY
; RUN: opt < %s -dfsan -dfsan-args-abi -S | FileCheck %s --check-prefix=ARGS_ABI
; RUN: opt < %s -dfsan -dfsan-fast-16-labels=true -S | FileCheck %s --check-prefix=FAST16
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define <4 x i4> @pass_vector(<4 x i4> %v) {
  ; ARGS_ABI: @"dfs$pass_vector"
  ; ARGS_ABI: ret { <4 x i4>, i16 }
  
  ; FAST16: @"dfs$pass_vector"
  ; FAST16: {{.*}} = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; FAST16: store i16 %1, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  ret <4 x i4> %v
}

define void @load_update_store_vector(<4 x i4>* %p) {
  ; FAST16: @"dfs$load_update_store_vector"
  ; FAST16: {{.*}} = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align 2

  %v = load <4 x i4>, <4 x i4>* %p
  %e2 = extractelement <4 x i4> %v, i32 2
  %v1 = insertelement <4 x i4> %v, i4 %e2, i32 0
  store <4 x i4> %v1, <4 x i4>* %p
  ret void
}

define <4 x i1> @icmp_vector(<4 x i8> %a, <4 x i8> %b) {
  ; LEGACY: @"dfs$icmp_vector"
  ; LEGACY: [[B:%.*]] = load i16, i16* inttoptr (i64 add (i64 ptrtoint ([100 x i64]* @__dfsan_arg_tls to i64), i64 2) to i16*), align [[ALIGN:2]]
  ; LEGACY: [[A:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; LEGACY: [[U:%.*]] = call zeroext i16 @__dfsan_union(i16 zeroext [[A]], i16 zeroext [[B]])
  ; LEGACY: [[PH:%.*]] = phi i16 [ [[U]], {{.*}} ], [ [[A]], {{.*}} ]
  ; LEGACY: store i16 [[PH]], i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  
  %r = icmp eq <4 x i8> %a, %b
  ret <4 x i1> %r
}

define <2 x i32> @const_vector() {
  ; LEGACY: @"dfs$const_vector"
  ; LEGACY: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
  
  ; FAST16: @"dfs$const_vector"
  ; FAST16: store i16 0, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align 2
  ret <2 x i32> < i32 42, i32 11 >
}

define <4 x i4> @call_vector(<4 x i4> %v) {
  ; LEGACY: @"dfs$call_vector"
  ; LEGACY: [[V:%.*]] = load i16, i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN:2]]
  ; LEGACY: store i16 [[V]], i16* bitcast ([100 x i64]* @__dfsan_arg_tls to i16*), align [[ALIGN]]
  ; LEGACY: %_dfsret = load i16, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]
  ; LEGACY: store i16 %_dfsret, i16* bitcast ([100 x i64]* @__dfsan_retval_tls to i16*), align [[ALIGN]]

  %r = call <4 x i4> @pass_vector(<4 x i4> %v)
  ret <4 x i4> %r
}


