;; Test that we don't add any instrumentation code to functions without
;; interesting memory accesses.
;
; RUN: opt < %s -passes='function(memprof),module(memprof-module)' -S -debug 2>&1 | FileCheck %s

;; Require asserts for -debug
; REQUIRES: asserts

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @_Z3foov() {
entry:
  ret void
}

;; Confirm we ran memprof and decided not to instrument
; CHECK: MEMPROF done instrumenting: 0 define void @_Z3foov

;; We should not add any instrumentation related code
; CHECK: define void @_Z3foov
; CHECK-NEXT: entry:
; CHECK-NEXT:  ret void
