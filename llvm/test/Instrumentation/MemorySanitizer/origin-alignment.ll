; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=1 -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS1 %s
; RUN: opt < %s -msan -msan-check-access-address=0 -msan-track-origins=2 -S | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS2 %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Check origin instrumentation of stores.
; Check that debug info for origin propagation code is set correctly.

@a8 = global i8 0, align 8
@a4 = global i8 0, align 4
@a2 = global i8 0, align 2
@a1 = global i8 0, align 1

; 8-aligned store => 8-aligned origin store, origin address is not realigned
define void @Store8(i8 %x) sanitize_memory {
entry:
  store i8 %x, i8* @a8, align 8
  ret void
}

; CHECK-LABEL: @Store8
; CHECK-ORIGINS1: [[ORIGIN:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS2: [[ORIGIN0:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS2: [[ORIGIN:%[01-9a-z]+]] = call i32 @__msan_chain_origin(i32 [[ORIGIN0]])
; CHECK: store i32 [[ORIGIN]], i32* inttoptr (i64 add (i64 xor (i64 ptrtoint (i8* @a8 to i64), i64 {{.*}}), i64 {{.*}}) to i32*), align 8
; CHECK: ret void


; 4-aligned store => 4-aligned origin store, origin address is not realigned
define void @Store4(i8 %x) sanitize_memory {
entry:
  store i8 %x, i8* @a4, align 4
  ret void
}

; CHECK-LABEL: @Store4
; CHECK-ORIGINS1: [[ORIGIN:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS2: [[ORIGIN0:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS2: [[ORIGIN:%[01-9a-z]+]] = call i32 @__msan_chain_origin(i32 [[ORIGIN0]])
; CHECK: store i32 [[ORIGIN]], i32* inttoptr (i64 add (i64 xor (i64 ptrtoint (i8* @a4 to i64), i64 {{.*}}), i64 {{.*}}) to i32*), align 4
; CHECK: ret void


; 2-aligned store => 4-aligned origin store, origin address is realigned
define void @Store2(i8 %x) sanitize_memory {
entry:
  store i8 %x, i8* @a2, align 2
  ret void
}

; CHECK-LABEL: @Store2
; CHECK-ORIGINS1: [[ORIGIN:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS2: [[ORIGIN0:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS2: [[ORIGIN:%[01-9a-z]+]] = call i32 @__msan_chain_origin(i32 [[ORIGIN0]])
; CHECK: store i32 [[ORIGIN]],  i32* inttoptr (i64 and (i64 add (i64 xor (i64 ptrtoint (i8* @a2 to i64), i64 {{.*}}), i64 {{.*}}), i64 -4) to i32*), align 4

; CHECK: ret void


; 1-aligned store => 4-aligned origin store, origin address is realigned
define void @Store1(i8 %x) sanitize_memory {
entry:
  store i8 %x, i8* @a1, align 1
  ret void
}

; CHECK-LABEL: @Store1
; CHECK-ORIGINS1: [[ORIGIN:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS2: [[ORIGIN0:%[01-9a-z]+]] = load {{.*}} @__msan_param_origin_tls
; CHECK-ORIGINS2: [[ORIGIN:%[01-9a-z]+]] = call i32 @__msan_chain_origin(i32 [[ORIGIN0]])
; CHECK: store i32 [[ORIGIN]], i32* inttoptr (i64 and (i64 add (i64 xor (i64 ptrtoint (i8* @a1 to i64), i64 {{.*}}), i64 {{.*}}), i64 -4) to i32*), align 4

; CHECK: ret void
