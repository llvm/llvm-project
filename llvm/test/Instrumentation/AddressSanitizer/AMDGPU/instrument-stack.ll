; This test checks that we are not instrumenting direct inbound stack accesses.
; RUN: opt < %s -passes=asan -asan-opt-stack -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-opt-stack -asan-mapping-scale=5 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;@sink = global ptr null, align 4

; Ignore direct inbounds stack access.
define void @foo() uwtable sanitize_address {
entry:
  %a = alloca i32, align 4
  store i32 42, ptr %a, align 4
  ret void
; CHECK-LABEL: define void @foo
; CHECK-NOT: __asan_report
; CHECK: ret void
}

; Don't ignore dynamic indexing.
define void @baz(i64 %i) sanitize_address {
entry:
  %a = alloca [10 x i32], align 4
  %e = getelementptr inbounds [10 x i32], ptr %a, i32 0, i64 %i
  store i32 42, ptr %e, align 4
  ret void
; CHECK-LABEL: define void @baz
; CHECK: __asan_report
; CHECK: ret void
}

define void @bar() sanitize_address {
entry:
  %a = alloca [10 x i32], align 4
  %e = getelementptr inbounds [10 x i32], ptr %a, i32 0, i64 12
  store i32 42, ptr %e, align 4
  ret void
; CHECK-LABEL: define void @bar
; CHECK: __asan_report
; CHECK: ret void
}

define void @endoftests() sanitize_address {
entry:
  ret void
; CHECK-LABEL: define void @endoftests
}

