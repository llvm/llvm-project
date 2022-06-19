; Test basic address sanitizer instrumentation.
;
; RUN: opt -passes='asan-pipeline' -S  < %s | FileCheck %s

target triple = "x86_64-pc-windows-msvc"
; CHECK: @llvm.global_ctors = {{.*}}@asan.module_ctor

define i32 @test_load(i32* %a) sanitize_address {
; First instrumentation in the function must be to load the dynamic shadow
; address into a local variable.
; CHECK-LABEL: @test_load
; CHECK: entry:
; CHECK-NEXT: %[[SHADOW:[^ ]*]] = load i64, i64* @__asan_shadow_memory_dynamic_address

; Shadow address is loaded and added into the whole offset computation.
; CHECK: add i64 %{{.*}}, %[[SHADOW]]

entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}

define i32 @__asan_options(i32* %a) sanitize_address {
; Asan functions are not instrumented. Asan function may be called by
; __asan_init before the shadow initialisation, which may lead to incorrect
; behavior of the instrumented code.
; CHECK-LABEL: @__asan_options
; CHECK: entry:
; CHECK-NEXT: %tmp1 = load i32, i32* %a, align 4
; CHECK-NEXT: ret i32 %tmp1

entry:
  %tmp1 = load i32, i32* %a, align 4
  ret i32 %tmp1
}
