; Test that AddressSanitizer instruments "(*a)++" only once.
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -S -asan-opt=1 | FileCheck %s -check-prefix=OPT1
; RUN: opt < %s -passes='asan-pipeline' -S -asan-opt=1 | FileCheck %s -check-prefix=OPT1
; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -S -asan-opt=0 | FileCheck %s -check-prefix=OPT0
; RUN: opt < %s -passes='asan-pipeline' -S -asan-opt=0 | FileCheck %s -check-prefix=OPT0

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
define void @IncrementMe(i32* %a) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  %tmp2 = add i32 %tmp1,  1
  store i32 %tmp2, i32* %a, align 4
  ret void
}

; With optimizations enabled we should see only one call to __asan_report_*
; OPT1: IncrementMe
; OPT1: __asan_report_
; OPT1-NOT: __asan_report_
; OPT1: ret void

; Without optimizations we should see two calls to __asan_report_*
; OPT0: IncrementMe
; OPT0: __asan_report_
; OPT0: __asan_report_
; OPT0: ret void
