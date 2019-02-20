; Verify that calls with !nosanitize are not instrumented by MSan.
; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan -S | FileCheck %s
; RUN: opt < %s -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan -msan-track-origins=1 -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @bar(i32 %x)

define void @foo() {
  call void @bar(i32 7), !nosanitize !{}
  ret void
}

; CHECK-LABEL: define void @foo
; CHECK-NOT: store {{.*}} @__msan_param_tls
; CHECK: call void @bar
; CHECK: ret void


@__sancov_gen_ = private global [1 x i8] zeroinitializer, section "__sancov_cntrs", align 1
define void @sancov() sanitize_memory {
entry:
  %0 = load i8, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__sancov_gen_, i64 0, i64 0), !nosanitize !{}
  %1 = add i8 %0, 1
  store i8 %1, i8* getelementptr inbounds ([1 x i8], [1 x i8]* @__sancov_gen_, i64 0, i64 0), !nosanitize !{}
  ret void
}

; CHECK-LABEL: define void @sancov
; CHECK-NOT: xor
; CHECK-NOT: 87960930222080
; CHECK: ret void


define void @load_store() sanitize_memory {
entry:
  %x = alloca i32, align 4, !nosanitize !{}
  store i32 4, i32* %x, align 4, !nosanitize !{}
  %0 = load i32, i32* %x, align 4, !nosanitize !{}
  %add = add nsw i32 %0, %0
  store i32 %add, i32* %x, align 4, !nosanitize !{}
  ret void
}

; CHECK-LABEL: define void @load_store
; CHECK-NOT: xor
; CHECK-NOT: 87960930222080
; CHECK: ret void
