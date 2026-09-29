; Test -msan-instrumentation-with-call-threshold
; Test that in with-calls mode there are no calls to __msan_chain_origin - they
; are done from __msan_maybe_store_origin_*.

; RUN: opt < %s -msan-check-access-address=0 -msan-instrumentation-with-call-threshold=0 -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan-check-access-address=0 -msan-instrumentation-with-call-threshold=0 -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS %s
; RUN: opt < %s -msan-check-access-address=0 -msan-instrumentation-with-call-threshold=0 -msan-track-origins=2 -S -passes=msan 2>&1 | FileCheck -check-prefix=CHECK -check-prefix=CHECK-ORIGINS %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @LoadAndCmp(ptr nocapture %a) nounwind uwtable sanitize_memory {
entry:
  %0 = load i32, ptr %a, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...) @foo() nounwind
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

declare void @foo(...)

; CHECK-LABEL: @LoadAndCmp
; CHECK: = load
; CHECK: = load
; CHECK: = zext i1 {{.*}} to i8
; CHECK: call void @__msan_maybe_warning_1(i8 zeroext {{.*}}, i32 zeroext {{.*}})
; CHECK-NOT: unreachable
; CHECK: ret void


define void @Store(ptr nocapture %p, i64 %x) nounwind uwtable sanitize_memory {
entry:
  store i64 %x, ptr %p, align 4
  ret void
}

; CHECK-LABEL: @Store
; CHECK: load {{.*}} @__msan_param_tls
; CHECK-ORIGINS: load {{.*}} @__msan_param_origin_tls
; CHECK: store
; CHECK-ORIGINS-NOT: __msan_chain_origin
; CHECK-ORIGINS: call void @__msan_maybe_store_origin_8(i64 zeroext {{.*}}, ptr {{.*}}, i32 zeroext {{.*}})
; CHECK-ORIGINS-NOT: __msan_chain_origin
; CHECK: store i64
; CHECK: ret void
