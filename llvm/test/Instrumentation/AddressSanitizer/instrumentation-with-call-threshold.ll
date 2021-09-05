; Test asan internal compiler flags:
;   -asan-instrumentation-with-call-threshold
;   -asan-memory-access-callback-prefix

; RUN: opt < %s -passes='asan-pipeline' -asan-instrumentation-with-call-threshold=1 -S | FileCheck %s --check-prefix=CHECK-CALL
; RUN: opt < %s -passes='asan-pipeline' -asan-instrumentation-with-call-threshold=0 -S | FileCheck %s --check-prefix=CHECK-CALL
; RUN: opt < %s -passes='asan-pipeline' -asan-instrumentation-with-call-threshold=0 -asan-memory-access-callback-prefix=__foo_ -S | FileCheck %s --check-prefix=CHECK-CUSTOM-PREFIX
; RUN: opt < %s -passes='asan-pipeline' -asan-instrumentation-with-call-threshold=5 -S | FileCheck %s --check-prefix=CHECK-INLINE
; RUN: opt < %s -passes='asan-pipeline'  -S | FileCheck %s --check-prefix=CHECK-INLINE
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

define void @test_load(i32* %a, i64* %b, i512* %c, i80* %d) sanitize_address {
entry:
; CHECK-CALL: call void @__asan_load4
; CHECK-CALL: call void @__asan_load8
; CHECK-CALL: call void @__asan_loadN{{.*}}i64 64)
; CHECK-CALL: call void @__asan_loadN{{.*}}i64 10)
; CHECK-CUSTOM-PREFIX: call void @__foo_load4
; CHECK-CUSTOM-PREFIX: call void @__foo_load8
; CHECK-CUSTOM-PREFIX: call void @__foo_loadN
; CHECK-INLINE-NOT: call void @__asan_load
  %tmp1 = load i32, i32* %a, align 4
  %tmp2 = load i64, i64* %b, align 8
  %tmp3 = load i512, i512* %c, align 32
  %tmp4 = load i80, i80* %d, align 8
  ret void
}


