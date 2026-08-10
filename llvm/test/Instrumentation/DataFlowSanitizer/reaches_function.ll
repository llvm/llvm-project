; RUN: opt < %s -passes=dfsan -dfsan-reaches-function-callbacks=1 -S | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare i32 @f()

define void @load(i32) {
  ; CHECK-LABEL: define void @load.dfsan
  ; CHECK: call{{.*}}@__dfsan_reaches_function_callback
  %i = alloca i32
  store i32 %0, ptr %i
  ret void
}

define void @store(ptr) {
  ; CHECK-LABEL: define void @store.dfsan
  ; CHECK: call{{.*}}@__dfsan_reaches_function_callback
  %load = load i32, ptr %0
  ret void
}

define void @call() {
  ; CHECK-LABEL: define void @call.dfsan
  ; CHECK: call{{.*}}@__dfsan_reaches_function_callback
  %ret = call i32 @f()
  ret void
}

; CHECK-LABEL: @__dfsan_reaches_function_callback(i8 zeroext, ptr, i32, ptr)
