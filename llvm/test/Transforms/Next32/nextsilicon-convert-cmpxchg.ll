; RUN: opt -S -passes=next-silicon-ir-builtins < %s | FileCheck %s

;; Confirm that the 128-bit cmpxchg instruction is converted to a call to
;; __ns_atomic_compare_exchange_16 and the arguments and the return value are
;; correctly transformed.

; ModuleID = 'test_lib.cpp'
source_filename = "test_lib.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: mustprogress noinline nounwind optnone ssp uwtable
define noundef zeroext i1 @_Z39opal_atomic_compare_exchange_strong_128PoS_o(ptr noundef %0, ptr noundef %1, i64 noundef %2, i64 noundef %3) #0 {
  %5 = alloca i128, align 16
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca i128, align 16
  %9 = alloca i128, align 16
  %10 = alloca i8, align 1
  %11 = getelementptr inbounds { i64, i64 }, ptr %5, i32 0, i32 0
  store i64 %2, ptr %11, align 16
  %12 = getelementptr inbounds { i64, i64 }, ptr %5, i32 0, i32 1
  store i64 %3, ptr %12, align 8
  %13 = load i128, ptr %5, align 16
  store ptr %0, ptr %6, align 8
  store ptr %1, ptr %7, align 8
  store i128 %13, ptr %8, align 16
  %14 = load ptr, ptr %6, align 8
  %15 = load ptr, ptr %7, align 8
  %16 = load i128, ptr %8, align 16
  store i128 %16, ptr %9, align 16
; CHECK-LABEL: store i128 %16, ptr %9, align 16
  %17 = load i128, ptr %15, align 16
; CHECK: [[CMP_VAL:%.*]] = load i128, ptr [[CMP_PTR:%.*]], align 16
  %18 = load i128, ptr %9, align 16
; CHECK-NEXT: [[NEW_VAL:%.*]] = load i128, ptr %9, align 16
; CHECK-NEXT: [[VAL_LOW:%.*]] = trunc i128 [[NEW_VAL]] to i64
; CHECK-NEXT: [[VALSHFT:%.*]] = ashr i128 [[NEW_VAL]], 64
; CHECK-NEXT: [[VAL_HIGH:%.*]] = trunc i128 [[VALSHFT]] to i64
  %19 = cmpxchg ptr %14, i128 %17, i128 %18 acquire monotonic, align 16
; CHECK-NEXT: [[CMP_RES:%.*]] = call i1 @__ns_atomic_compare_exchange_16(ptr [[PTR1:%.*]], ptr [[CMP_PTR]], i64 [[VAL_LOW]], i64 [[VAL_HIGH]], i32 1, i32 0)
; CHECK-NEXT: [[LOAD_VAL:%.*]] = load i128, ptr [[CMP_PTR]], align 8
; CHECK-NEXT: [[INS_VAL:%.*]] = insertvalue { i128, i1 } poison, i128 [[LOAD_VAL]], 0
; CHECK-NEXT: [[INS_CMP:%.*]] = insertvalue { i128, i1 } poison, i1 [[CMP_RES]], 1
  %20 = extractvalue { i128, i1 } %19, 0
; CHECK-NEXT: [[VALUE:%.*]] = extractvalue { i128, i1 } poison, 0
  %21 = extractvalue { i128, i1 } %19, 1
; CHECK-NEXT: [[SUCCESS:%.*]] = extractvalue { i128, i1 } poison, 1
  br i1 %21, label %23, label %22
; CHECK-NEXT: br i1 [[SUCCESS]], label %29, label %28

22:                                               ; preds = %4
  store i128 %20, ptr %15, align 16
; CHECK: store i128 [[VALUE]], ptr [[CMP_PTR]], align 16
  br label %23

23:                                               ; preds = %22, %4
  %24 = zext i1 %21 to i8
; CHECK: [[ZEXT:%.*]] = zext i1 [[SUCCESS]] to i8
  store i8 %24, ptr %10, align 1
  %25 = load i8, ptr %10, align 1
  %26 = trunc i8 %25 to i1
  ret i1 %26
}

attributes #0 = { mustprogress noinline nounwind optnone ssp uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="core2" "target-features"="+cmov,+cx16,+cx8,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+ssse3,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{i32 7, !"frame-pointer", i32 2}
!4 = !{!"NextSilicon clang version 17.0.6 (git@github.com:nextsilicon/next-llvm-project.git bf13f0d0b1e914d4e621581a24330945bf31b1d2)"}
