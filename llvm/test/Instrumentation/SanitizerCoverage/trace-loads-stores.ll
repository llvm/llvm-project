; Test -sanitizer-coverage-inline-8bit-counters=1
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-loads=1  -S | FileCheck %s --check-prefix=LOADS
; RUN: opt < %s -passes='module(sancov-module)' -sanitizer-coverage-level=1 -sanitizer-coverage-trace-stores=1  -S | FileCheck %s --check-prefix=STORES

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @foo(i8* %p1, i16* %p2, i32* %p4, i64* %p8, i128* %p16) {
; =================== loads
  %1 = load i8, i8* %p1
  %2 = load i16, i16* %p2
  %3 = load i32, i32* %p4
  %4 = load i64, i64* %p8
  %5 = load i128, i128* %p16
; LOADS: call void @__sanitizer_cov_load1(i8* %p1)
; LOADS: call void @__sanitizer_cov_load2(i16* %p2)
; LOADS: call void @__sanitizer_cov_load4(i32* %p4)
; LOADS: call void @__sanitizer_cov_load8(i64* %p8)
; LOADS: call void @__sanitizer_cov_load16(i128* %p16)

; =================== stores
  store i8   %1, i8*   %p1
  store i16  %2, i16*  %p2
  store i32  %3, i32*  %p4
  store i64  %4, i64*  %p8
  store i128 %5, i128* %p16
; STORES: call void @__sanitizer_cov_store1(i8* %p1)
; STORES: call void @__sanitizer_cov_store2(i16* %p2)
; STORES: call void @__sanitizer_cov_store4(i32* %p4)
; STORES: call void @__sanitizer_cov_store8(i64* %p8)
; STORES: call void @__sanitizer_cov_store16(i128* %p16)

  ret void
}
