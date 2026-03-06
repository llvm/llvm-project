; Test using dynamic shadow address on Windows
;
; Only x86_64 and aarch64 Windows should use dynamic shadow.
; RUN: opt -passes=asan -mtriple=x86_64-pc-windows-msvc -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC
; RUN: opt -passes=asan -mtriple=aarch64-pc-windows-msvc -S < %s | FileCheck %s --check-prefixes=CHECK,CHECK-DYNAMIC

define i32 @test_load(ptr %a) sanitize_address {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_load
; CHECK: entry:
; CHECK-DYNAMIC-NEXT: %[[SHADOW:[^ ]*]] = load i64, ptr @__asan_shadow_memory_dynamic_address

; Shadow address is loaded and added into the whole offset computation.
; CHECK-DYNAMIC: add i64 %{{.*}}, %[[SHADOW]]

entry:
  %tmp1 = load i32, ptr %a, align 4
  ret i32 %tmp1
}
