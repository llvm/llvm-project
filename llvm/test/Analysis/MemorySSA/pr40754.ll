; RUN: opt -licm -verify-memoryssa -S < %s | FileCheck %s
; REQUIRES: asserts

target datalayout = "E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-v128:64-a:8:16-n32:64"
target triple = "systemz-unknown"

@g_120 = external dso_local local_unnamed_addr global [8 x [4 x [6 x i32]]], align 4
@g_185 = external dso_local local_unnamed_addr global i32, align 4
@g_329 = external dso_local local_unnamed_addr global i16, align 2

; Function Attrs: norecurse noreturn nounwind
define dso_local void @func_65() local_unnamed_addr {
; CHECK-LABEL: @func_65()
label0:
  br label %label1

label1:                                      ; preds = %.thread, %label0
  br label %label2

label2:                                      ; preds = %.critedge, %label1
  br label %label3

label3:                                      ; preds = %label5, %label2
  %storemerge = phi i32 [ 0, %label2 ], [ %tmp6, %label5 ]
  store i32 %storemerge, i32* @g_185, align 4
  %tmp4 = icmp ult i32 %storemerge, 2
  br i1 %tmp4, label %label5, label %.thread.loopexit

label5:                                      ; preds = %label3
  %tmp6 = add i32 %storemerge, 1
  %tmp7 = zext i32 %tmp6 to i64
  %tmp8 = getelementptr [8 x [4 x [6 x i32]]], [8 x [4 x [6 x i32]]]* @g_120, i64 0, i64 undef, i64 %tmp7, i64 undef
  %tmp9 = load i32, i32* %tmp8, align 4
  %tmp10 = icmp eq i32 %tmp9, 0
  br i1 %tmp10, label %label3, label %label11

label11:                                     ; preds = %label5
  %storemerge.lcssa4 = phi i32 [ %storemerge, %label5 ]
  %tmp12 = icmp eq i32 %storemerge.lcssa4, 0
  br i1 %tmp12, label %.critedge, label %.thread.loopexit3

.critedge:                                        ; preds = %label11
  store i16 0, i16* @g_329, align 2
  br label %label2

.thread.loopexit:                                 ; preds = %label3
  br label %.thread

.thread.loopexit3:                                ; preds = %label11
  br label %.thread

.thread:                                          ; preds = %.thread.loopexit3, %.thread.loopexit
  br label %label1
}

