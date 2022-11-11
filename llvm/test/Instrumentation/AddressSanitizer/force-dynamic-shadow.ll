; Test -asan-force-dynamic-shadow flag.
;
; RUN: opt -passes=asan -S -asan-force-dynamic-shadow=1 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-FDS
; RUN: opt -passes=asan -S -asan-force-dynamic-shadow=0 < %s | FileCheck %s --check-prefixes=CHECK,CHECK-NDS

target triple = "x86_64-unknown-linux-gnu"

define i32 @test_load(i32* %a) sanitize_address {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_load
; CHECK: entry:
; CHECK-FDS-NEXT: %[[SHADOW:[^ ]*]] = load i64, i64* @__asan_shadow_memory_dynamic_address
; CHECK-NDS-NOT: __asan_shadow_memory_dynamic_address

; Shadow address is loaded and added into the whole offset computation.
; CHECK-FDS: add i64 %{{.*}}, %[[SHADOW]]

entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
