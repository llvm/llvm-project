; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @func() sanitize_memory {
entry:
  %0 = alloca i32, i32 0, align 4
  %1 = alloca i32, i32 0, align 4
  %2 = alloca i32, i32 0, align 4
  %3 = alloca i32, i32 0, align 4
  %4 = alloca i32, i32 0, align 4
  %5 = alloca i32, i32 0, align 4
  %6 = alloca i32, i32 0, align 4
  %7 = alloca i32, i32 0, align 4
  %8 = alloca i32, i32 0, align 4
  %9 = alloca i32, i32 0, align 4
  %10 = alloca i32, i32 0, align 4
  %11 = alloca i32, i32 0, align 4
  %12 = alloca i32, i32 0, align 4
  %13 = alloca i32, i32 0, align 4
  %14 = alloca i32, i32 0, align 4
  %15 = alloca i32, i32 0, align 4
  %16 = alloca i32, i32 0, align 4
  %17 = alloca i32, i32 0, align 4
  %18 = alloca i32, i32 0, align 4
  %19 = alloca i32, i32 0, align 4
  %20 = alloca i32, i32 0, align 4
  ret void
}

; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @0, ptr @1)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @2, ptr @3)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @4, ptr @5)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @6, ptr @7)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @8, ptr @9)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @10, ptr @11)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @12, ptr @13)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @14, ptr @15)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @16, ptr @17)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @18, ptr @19)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @20, ptr @21)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @22, ptr @23)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @24, ptr @25)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @26, ptr @27)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @28, ptr @29)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @30, ptr @31)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @32, ptr @33)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @34, ptr @35)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @36, ptr @37)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @38, ptr @39)
; CHECK: call void @__msan_set_alloca_origin_with_descr(ptr %{{[0-9]+}}, i64 0, ptr @40, ptr @41)
