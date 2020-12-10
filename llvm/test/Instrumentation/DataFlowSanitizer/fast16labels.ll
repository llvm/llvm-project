; Test that -dfsan-fast-16-labels mode uses inline ORs rather than calling
; __dfsan_union or __dfsan_union_load.
; RUN: opt < %s -dfsan -dfsan-fast-16-labels -S | FileCheck %s --implicit-check-not="call{{.*}}__dfsan_union"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i8 @add(i8 %a, i8 %b) {
  ; CHECK-LABEL: define i8 @"dfs$add"
  ; CHECK-DAG: %[[ALABEL:.*]] = load [[ST:.*]], [[ST]]* bitcast ([[VT:\[.*\]]]* @__dfsan_arg_tls to [[ST]]*), align [[ALIGN:2]]
  ; CHECK-DAG: %[[BLABEL:.*]] = load [[ST]], [[ST]]* inttoptr (i64 add (i64 ptrtoint ([[VT]]* @__dfsan_arg_tls to i64), i64 2) to [[ST]]*), align [[ALIGN]]
  ; CHECK: %[[ADDLABEL:.*]] = or i16 %[[ALABEL]], %[[BLABEL]]
  ; CHECK: add i8
  ; CHECK: store [[ST]] %[[ADDLABEL]], [[ST]]* bitcast ([[VT]]* @__dfsan_retval_tls to [[ST]]*), align [[ALIGN]]
  ; CHECK: ret i8
  %c = add i8 %a, %b
  ret i8 %c
}

define i8 @load8(i8* %p) {
  ; CHECK-LABEL: define i8 @"dfs$load8"
  ; CHECK: load i16, i16*
  ; CHECK: ptrtoint i8* {{.*}} to i64
  ; CHECK: and i64
  ; CHECK: mul i64
  ; CHECK: inttoptr i64
  ; CHECK: load i16, i16*
  ; CHECK: or i16
  ; CHECK: load i8, i8*
  ; CHECK: store i16 {{.*}} @__dfsan_retval_tls
  ; CHECK: ret i8

  %a = load i8, i8* %p
  ret i8 %a
}

define i16 @load16(i16* %p) {
  ; CHECK-LABEL: define i16 @"dfs$load16"
  ; CHECK: ptrtoint i16*
  ; CHECK: and i64
  ; CHECK: mul i64
  ; CHECK: inttoptr i64 {{.*}} i16*
  ; CHECK: getelementptr i16
  ; CHECK: load i16, i16*
  ; CHECK: load i16, i16*
  ; CHECK: or i16
  ; CHECK: or i16
  ; CHECK: load i16, i16*
  ; CHECK: store {{.*}} @__dfsan_retval_tls
  ; CHECK: ret i16

  %a = load i16, i16* %p
  ret i16 %a
}

define i32 @load32(i32* %p) {
  ; CHECK-LABEL: define i32 @"dfs$load32"
  ; CHECK: ptrtoint i32*
  ; CHECK: and i64
  ; CHECK: mul i64
  ; CHECK: inttoptr i64 {{.*}} i16*
  ; CHECK: bitcast i16* {{.*}} i64*
  ; CHECK: load i64, i64*
  ; CHECK: lshr i64 {{.*}}, 32
  ; CHECK: or i64
  ; CHECK: lshr i64 {{.*}}, 16
  ; CHECK: or i64
  ; CHECK: trunc i64 {{.*}} i16
  ; CHECK: or i16
  ; CHECK: load i32, i32*
  ; CHECK: store i16 {{.*}} @__dfsan_retval_tls
  ; CHECK: ret i32

  %a = load i32, i32* %p
  ret i32 %a
}

define i64 @load64(i64* %p) {
  ; CHECK-LABEL: define i64 @"dfs$load64"
  ; CHECK: ptrtoint i64*
  ; CHECK: and i64
  ; CHECK: mul i64
  ; CHECK: inttoptr i64 {{.*}} i16*
  ; CHECK: bitcast i16* {{.*}} i64*
  ; CHECK: load i64, i64*
  ; CHECK: getelementptr i64, i64* {{.*}}, i64 1
  ; CHECK: load i64, i64*
  ; CHECK: or i64
  ; CHECK: lshr i64 {{.*}}, 32
  ; CHECK: or i64
  ; CHECK: lshr i64 {{.*}}, 16
  ; CHECK: or i64
  ; CHECK: trunc i64 {{.*}} i16
  ; CHECK: or i16
  ; CHECK: load i64, i64*
  ; CHECK: store i16 {{.*}} @__dfsan_retval_tls
  ; CHECK: ret i64

  %a = load i64, i64* %p
  ret i64 %a
}
