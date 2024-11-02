; RUN: opt -S -passes=globalopt -o - < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: [[_ZL3G_I_INIT:@.*]] = internal unnamed_addr global i1 false
@_ZL3g_i = internal global i32* null, align 8
@.str = private unnamed_addr constant [2 x i8] c"0\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c"1\00", align 1

; CHECK-LABEL: define {{.*}} @main(
; CHECK-NEXT:    store i1 false, i1* [[_ZL3G_I_INIT]], align 1
define dso_local i32 @main() {
  store i32* null, i32** @_ZL3g_i, align 8
  call void @_ZL13PutsSomethingv()
  ret i32 0
}

; CHECK-LABEL: define {{.*}} @_ZL13PutsSomethingv()
; CHECK-NEXT:    [[_ZL3G_I_INIT_VAL:%.*]] = load i1, i1* [[_ZL3G_I_INIT]], align 1
; CHECK-NEXT:    [[NOTINIT:%.*]] = xor i1 [[_ZL3G_I_INIT_VAL]], true
; CHECK-NEXT:    br i1 [[NOTINIT]], label %[[TMP1:.*]], label %[[TMP3:.*]]
; CHECK:       [[TMP1]]:
; CHECK-NEXT:    store i1 true, i1* [[_ZL3G_I_INIT]], align 1
define internal void @_ZL13PutsSomethingv() {
  %1 = load i32*, i32** @_ZL3g_i, align 8
  %2 = icmp eq i32* %1, null
  br i1 %2, label %3, label %7

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

declare dso_local noalias i8* @malloc(i64) inaccessiblememonly allockind("alloc,uninitialized") allocsize(0)

declare dso_local i32 @puts(i8* nocapture readonly)
