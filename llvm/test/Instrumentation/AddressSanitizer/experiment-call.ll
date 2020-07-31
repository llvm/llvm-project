; Test optimization experiments.
; -asan-force-experiment flag turns all memory accesses into experiments.
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -asan-force-experiment=42 -asan-instrumentation-with-call-threshold=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -asan-force-experiment=42 -asan-instrumentation-with-call-threshold=0 -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @load1(i8* %p) sanitize_address {
entry:
  %t = load i8, i8* %p, align 1
  ret void
; CHECK-LABEL: define void @load1
; CHECK: __asan_exp_load1{{.*}} i32 42
; CHECK: ret void
}

define void @load2(i16* %p) sanitize_address {
entry:
  %t = load i16, i16* %p, align 2
  ret void
; CHECK-LABEL: define void @load2
; CHECK: __asan_exp_load2{{.*}} i32 42
; CHECK: ret void
}

define void @load4(i32* %p) sanitize_address {
entry:
  %t = load i32, i32* %p, align 4
  ret void
; CHECK-LABEL: define void @load4
; CHECK: __asan_exp_load4{{.*}} i32 42
; CHECK: ret void
}

define void @load8(i64* %p) sanitize_address {
entry:
  %t = load i64, i64* %p, align 8
  ret void
; CHECK-LABEL: define void @load8
; CHECK: __asan_exp_load8{{.*}} i32 42
; CHECK: ret void
}

define void @load16(i128* %p) sanitize_address {
entry:
  %t = load i128, i128* %p, align 16
  ret void
; CHECK-LABEL: define void @load16
; CHECK: __asan_exp_load16{{.*}} i32 42
; CHECK: ret void
}

define void @loadN(i48* %p) sanitize_address {
entry:
  %t = load i48, i48* %p, align 1
  ret void
; CHECK-LABEL: define void @loadN
; CHECK: __asan_exp_loadN{{.*}} i32 42
; CHECK: ret void
}

define void @store1(i8* %p) sanitize_address {
entry:
  store i8 1, i8* %p, align 1
  ret void
; CHECK-LABEL: define void @store1
; CHECK: __asan_exp_store1{{.*}} i32 42
; CHECK: ret void
}

define void @store2(i16* %p) sanitize_address {
entry:
  store i16 1, i16* %p, align 2
  ret void
; CHECK-LABEL: define void @store2
; CHECK: __asan_exp_store2{{.*}} i32 42
; CHECK: ret void
}

define void @store4(i32* %p) sanitize_address {
entry:
  store i32 1, i32* %p, align 4
  ret void
; CHECK-LABEL: define void @store4
; CHECK: __asan_exp_store4{{.*}} i32 42
; CHECK: ret void
}

define void @store8(i64* %p) sanitize_address {
entry:
  store i64 1, i64* %p, align 8
  ret void
; CHECK-LABEL: define void @store8
; CHECK: __asan_exp_store8{{.*}} i32 42
; CHECK: ret void
}

define void @store16(i128* %p) sanitize_address {
entry:
  store i128 1, i128* %p, align 16
  ret void
; CHECK-LABEL: define void @store16
; CHECK: __asan_exp_store16{{.*}} i32 42
; CHECK: ret void
}

define void @storeN(i48* %p) sanitize_address {
entry:
  store i48 1, i48* %p, align 1
  ret void
; CHECK-LABEL: define void @storeN
; CHECK: __asan_exp_storeN{{.*}} i32 42
; CHECK: ret void
}
