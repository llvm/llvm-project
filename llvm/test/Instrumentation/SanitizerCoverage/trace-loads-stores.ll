; Test -sanitizer-coverage-inline-8bit-counters=1
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-loads=1  -S | FileCheck %s --check-prefix=LOADS
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-stores=1  -S | FileCheck %s --check-prefix=STORES

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(ptr %p1, ptr %p2, ptr %p4, ptr %p8, ptr %p16) {
; =================== loads
  %1 = load i8, ptr %p1
  %2 = load i16, ptr %p2
  %3 = load i32, ptr %p4
  %4 = load i64, ptr %p8
  %5 = load i128, ptr %p16
; LOADS: call void @__sanitizer_cov_load1(ptr %p1)
; LOADS: call void @__sanitizer_cov_load2(ptr %p2)
; LOADS: call void @__sanitizer_cov_load4(ptr %p4)
; LOADS: call void @__sanitizer_cov_load8(ptr %p8)
; LOADS: call void @__sanitizer_cov_load16(ptr %p16)

; =================== stores
  store i8   %1, ptr   %p1
  store i16  %2, ptr  %p2
  store i32  %3, ptr  %p4
  store i64  %4, ptr  %p8
  store i128 %5, ptr %p16
; STORES: call void @__sanitizer_cov_store1(ptr %p1)
; STORES: call void @__sanitizer_cov_store2(ptr %p2)
; STORES: call void @__sanitizer_cov_store4(ptr %p4)
; STORES: call void @__sanitizer_cov_store8(ptr %p8)
; STORES: call void @__sanitizer_cov_store16(ptr %p16)

  ret void
}
