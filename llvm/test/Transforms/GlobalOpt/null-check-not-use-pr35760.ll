; RUN: opt -S -globalopt -o - < %s | FileCheck %s

; No malloc promotion with non-null check.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@_ZL3g_i = internal global i32* null, align 8
@_ZL3g_j = global i32* null, align 8
@.str = private unnamed_addr constant [2 x i8] c"0\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"1\00", align 1

define dso_local i32 @main() {
  store i32* null, i32** @_ZL3g_i, align 8
  call void @_ZL13PutsSomethingv()
  ret i32 0
}

; CHECK-LABEL: define {{.*}} @_ZL13PutsSomethingv()
; CHECK-NEXT:    [[TMP1:%.*]] = load i32*, i32** @_ZL3g_i, align 8
; CHECK-NEXT:    [[TMP2:%.*]] = load i32*, i32** @_ZL3g_j, align 8
; CHECK-NEXT:    icmp eq i32* [[TMP1]], [[TMP2]]
define internal void @_ZL13PutsSomethingv() {
  %1 = load i32*, i32** @_ZL3g_i, align 8
  %2 = load i32*, i32** @_ZL3g_j, align 8
  %cmp = icmp eq i32* %1, %2
  br i1 %cmp, label %3, label %7

3:                                                ; preds = %0
  %4 = call noalias i8* @malloc(i64 4) #3
  %5 = bitcast i8* %4 to i32*
  store i32* %5, i32** @_ZL3g_i, align 8
  %6 = call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str, i64 0, i64 0))
  br label %9

7:                                                ; preds = %0
  %8 = call i32 @puts(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.1, i64 0, i64 0))
  br label %9

9:                                                ; preds = %7, %3
  ret void
}

declare dso_local noalias i8* @malloc(i64)

declare dso_local i32 @puts(i8* nocapture readonly)
