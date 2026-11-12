; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S -passes=msan 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-ORIGIN
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @__atomic_load(i64, ptr, ptr, i32)
declare void @__atomic_store(i64, ptr, ptr, i32)

define i24 @odd_sized_load(ptr %ptr) sanitize_memory {
; CHECK: @odd_sized_load(ptr {{.*}}[[PTR:%.+]])
; CHECK: [[VAL_PTR:%.*]] = alloca i24, align 1
; CHECK-ORIGIN: @__msan_set_alloca_origin
; CHECK: call void @__atomic_load(i64 3, ptr [[PTR]], ptr [[VAL_PTR]], i32 2)

; CHECK: ptrtoint ptr [[PTR]]
; CHECK: xor
; CHECK: [[SPTR_I8:%.*]] = inttoptr
; CHECK-ORIGIN: add
; CHECK-ORIGIN: and
; CHECK-ORIGIN: [[OPTR:%.*]] = inttoptr

; CHECK: ptrtoint ptr [[VAL_PTR]]
; CHECK: xor
; CHECK: [[VAL_SPTR_I8:%.*]] = inttoptr
; CHECK-ORIGIN: add
; CHECK-ORIGIN: and
; CHECK-ORIGIN: [[VAL_OPTR:%.*]] = inttoptr

; CHECK: call void @llvm.memcpy{{.*}}(ptr align 1 [[VAL_SPTR_I8]], ptr align 1 [[SPTR_I8]], i64 3

; CHECK-ORIGIN: [[ARG_ORIGIN:%.*]] = load i32, ptr [[OPTR]]
; CHECK-ORIGIN: [[VAL_ORIGIN:%.*]] = call i32 @__msan_chain_origin(i32 [[ARG_ORIGIN]])
; CHECK-ORIGIN: call void @__msan_set_origin(ptr [[VAL_PTR]], i64 3, i32 [[VAL_ORIGIN]])

; CHECK: [[VAL:%.*]] = load i24, ptr [[VAL_PTR]]
; CHECK: ret i24 [[VAL]]
  %val_ptr = alloca i24, align 1
  call void @__atomic_load(i64 3, ptr %ptr, ptr %val_ptr, i32 0)
  %val = load i24, ptr %val_ptr
  ret i24 %val
}

define void @odd_sized_store(ptr %ptr, i24 %val) sanitize_memory {
; CHECK: @odd_sized_store(ptr {{.*}}[[PTR:%.+]], i24 {{.*}}[[VAL:%.+]])
; CHECK: [[VAL_PTR:%.*]] = alloca i24, align 1
; CHECK: store i24 [[VAL]], ptr [[VAL_PTR]]

; CHECK: ptrtoint ptr [[PTR]]
; CHECK: xor
; CHECK: [[SPTR_I8:%.*]] = inttoptr
; CHECK: call void @llvm.memset{{.*}}(ptr align 1 [[SPTR_I8]], i8 0, i64 3

; CHECK: call void @__atomic_store(i64 3, ptr [[VAL_PTR]], ptr [[PTR]], i32 3)
; CHECK: ret void
  %val_ptr = alloca i24, align 1
  store i24 %val, ptr %val_ptr
  call void @__atomic_store(i64 3, ptr %val_ptr, ptr %ptr, i32 0)
  ret void
}

