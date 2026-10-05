; RUN: opt < %s -S -passes=msan 2>&1 | FileCheck %s

; Region instrumentation is a KernelMemorySanitizer feature: it refers to the
; context state returned by __msan_get_context_state(), which only exists in the
; kernel runtime. Under userspace MemorySanitizer the markers must simply be
; ignored - this used to crash the compiler on a null MsanGetContextStateFn.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()

define i32 @region_in_userspace_msan(ptr %p) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  %v = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret i32 %v
}

; CHECK-LABEL: @region_in_userspace_msan(
; CHECK-NOT: __msan_get_context_state
; CHECK-NOT: msan_context_state
; CHECK: %v = load i32, ptr %p
; CHECK: ret i32 %v
