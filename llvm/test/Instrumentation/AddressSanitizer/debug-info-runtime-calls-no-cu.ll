; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-emit-debug-info -S | FileCheck %s --implicit-check-not='declare !dbg'

define void @test(ptr %src) sanitize_address {
entry:
  %0 = load i32, ptr %src, align 4
  ret void
}

; CHECK: call void @__asan_load4(i64
; CHECK: declare void @__asan_load4(i64)
