; Test asan internal compiler flags:
;   -asan-recover=1

; RUN: opt < %s -passes=asan -asan-recover -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo(i32* %p) sanitize_address {
; CHECK: __asan_report_load4_noabort
; CHECK-NOT: unreachable
  %1 = load i32, i32* %p, align 4
  ret i32 %1
}

