; RUN: opt -licm -enable-mssa-loop-dependency -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "systemz-unknown"

@g_120 = external dso_local local_unnamed_addr global [8 x [4 x [6 x i32]]], align 4
@g_185 = external dso_local local_unnamed_addr global i32, align 4
@g_329 = external dso_local local_unnamed_addr global i16, align 2

; Function Attrs: norecurse noreturn nounwind
define dso_local void @func_65() local_unnamed_addr {
; CHECK-LABEL: @func_65()
  br label %1

; <label>:1:                                      ; preds = %.thread, %0
  br label %2

; <label>:2:                                      ; preds = %.critedge, %1
  br label %3

; <label>:3:                                      ; preds = %5, %2
  %storemerge = phi i32 [ 0, %2 ], [ %6, %5 ]
  store i32 %storemerge, i32* @g_185, align 4
  %4 = icmp ult i32 %storemerge, 2
  br i1 %4, label %5, label %.thread.loopexit

; <label>:5:                                      ; preds = %3
  %6 = add i32 %storemerge, 1
  %7 = zext i32 %6 to i64
  %8 = getelementptr [8 x [4 x [6 x i32]]], [8 x [4 x [6 x i32]]]* @g_120, i64 0, i64 undef, i64 %7, i64 undef
  %9 = load i32, i32* %8, align 4
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %3, label %11

; <label>:11:                                     ; preds = %5
  %storemerge.lcssa4 = phi i32 [ %storemerge, %5 ]
  %12 = icmp eq i32 %storemerge.lcssa4, 0
  br i1 %12, label %.critedge, label %.thread.loopexit3

.critedge:                                        ; preds = %11
  store i16 0, i16* @g_329, align 2
  br label %2

.thread.loopexit:                                 ; preds = %3
  br label %.thread

.thread.loopexit3:                                ; preds = %11
  br label %.thread

.thread:                                          ; preds = %.thread.loopexit3, %.thread.loopexit
  br label %1
}

