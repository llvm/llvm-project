; Test kernel inline hwasan instrumentation.

; RUN: opt < %s -passes=asan-pipeline -asan-kernel=1 -asan-recover=1 -asan-instrumentation-with-call-threshold=10000 -S | FileCheck --check-prefixes=CHECK-INLINE %s
; RUN: opt < %s -passes=asan-pipeline -asan-kernel=1 -asan-recover=1 -asan-instrumentation-with-call-threshold=0 -S | FileCheck --check-prefixes=CHECK-CALLBACK %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @test_load(i32* %a, i64* %b, i512* %c, i80* %d) sanitize_address {
entry:
  %tmp1 = load i32, i32* %a, align 4
  %tmp2 = load i64, i64* %b, align 8
  %tmp3 = load i512, i512* %c, align 32
  %tmp4 = load i80, i80* %d, align 8
  ret void
}
; CHECK-INLINE: call void @__asan_report_load4_noabort
; CHECK-INLINE: call void @__asan_report_load8_noabort
; CHECK-INLINE: call void @__asan_report_load_n_noabort
; CHECK-INLINE-NOT: call void @__asan_load4_noabort
; CHECK-INLINE-NOT: call void @__asan_load8_noabort
; CHECK-INLINE-NOT: call void @__asan_loadN_noabort
; CHECK-CALLBACK: call void @__asan_load4_noabort
; CHECK-CALLBACK: call void @__asan_load8_noabort
; CHECK-CALLBACK: call void @__asan_loadN_noabort
; CHECK-CALLBACK-NOT: call void @__asan_report_load4_noabort
; CHECK-CALLBACK-NOT: call void @__asan_report_load8_noabort
; CHECK-CALLBACK-NOT: call void @__asan_report_load_n_noabort
