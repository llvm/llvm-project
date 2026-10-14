; RUN: opt < %s -passes=dfsan -dfsan-track-origins=1 -S | FileCheck %s

target datalayout = "E-p:64:64:64-i1:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-n8:16:32:64-S128"
target triple = "s390x-unknown-linux-gnu"

define i64 @load64(ptr %p) {
  ; CHECK-LABEL: @load64.dfsan
  ; CHECK: %[[SHADOW:.*]] = load i64, ptr {{.*}}, align 1
  ; CHECK-NEXT: and i64 %[[SHADOW]], -4294967296

  %a = load i64, ptr %p
  ret i64 %a
}
