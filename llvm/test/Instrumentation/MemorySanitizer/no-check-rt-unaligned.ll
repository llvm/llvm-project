; RUN: opt < %s -msan-eager-checks -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan-eager-checks -msan-check-access-address=0 -msan-track-origins=2 -S -passes=msan 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-ORIGIN
; RUN: opt < %s -msan-eager-checks -msan -msan-check-access-address=0 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare noundef i8 @__sanitizer_unaligned_load8(i8* noundef)
declare void @__sanitizer_unaligned_store8(i8* noundef, i8 noundef)

define noundef i8 @unaligned_load(i8* noundef %ptr) sanitize_memory {
; CHECK: @unaligned_load(i8* {{.*}}[[PTR:%.+]])
; CHECK: store i64 0, {{.*}} @__msan_param_tls
; CHECK: [[VAL:%.*]] = call noundef i8 @__sanitizer_unaligned_load8(i8* noundef [[PTR]])
  %val = call noundef i8 @__sanitizer_unaligned_load8(i8* noundef %ptr)
; CHECK: load {{.*}} @__msan_retval_tls
; CHECK-ORIGIN: load {{.*}} @__msan_retval_origin_tls
; CHECK: call void @__msan_warning_{{.*}}noreturn
; CHECK: ret i8 [[VAL]]
  ret i8 %val
}

define void @unaligned_store(i8* noundef %ptr, i8 noundef %val) sanitize_memory {
; CHECK: @unaligned_store(i8* {{.*}}[[PTR:%.+]], i8 {{.*}}[[VAL:%.+]])
; CHECK: store i64 0, {{.*}} @__msan_param_tls
; CHECK: store i8 0, {{.*}} @__msan_param_tls
; CHECK: call void @__sanitizer_unaligned_store8(i8* noundef [[PTR]], i8 noundef [[VAL]])
  call void @__sanitizer_unaligned_store8(i8* noundef %ptr, i8 noundef %val)
; CHECK: ret void
  ret void
}
