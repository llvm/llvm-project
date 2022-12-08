; RUN: opt < %s -passes=inline -inline-threshold=0 -debug-only=inline-cost -print-instruction-comments -S -mtriple=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -inline-threshold=0 -debug-only=inline-cost -print-instruction-comments -S -mtriple=x86_64-unknown-linux-gnu 2>&1 | FileCheck %s
; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-unknown"

; Check that insertvalue's aren't free.

; CHECK: Analyzing call of callee... (caller:caller_range)
; CHECK-NEXT: Initial cost: -40
; CHECK-NEXT: define { i32, i32 } @callee({ i32, i32 } %arg, i32 %arg1) {
; CHECK-NEXT: ; cost before = -40, cost after = -35, threshold before = 0, threshold after = 0, cost delta = 5
; CHECK-NEXT:   %r = insertvalue { i32, i32 } %arg, i32 %arg1, 0
; CHECK-NEXT: ; cost before = -35, cost after = -35, threshold before = 0, threshold after = 0, cost delta = 0
; CHECK-NEXT:   ret { i32, i32 } %r
; CHECK-NEXT: }

define {i32, i32} @callee({i32, i32} %arg, i32 %arg1) {
  %r = insertvalue {i32, i32} %arg, i32 %arg1, 0
  ret {i32, i32} %r
}

define {i32, i32} @caller_range({i32, i32} %arg, i32 %arg1) {
  %r = call {i32, i32} @callee({i32, i32} %arg, i32 %arg1)
  ret {i32, i32} %r
}
