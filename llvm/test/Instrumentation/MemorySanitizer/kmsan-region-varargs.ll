; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s

; VarArgHelpers emit their setup into the function prologue. In a
; region-instrumented function the prologue is outside every region, so
; %msan_context_state has not been stored to yet: the emitted code used to load
; that slot and then read and write through the resulting pointer. Varargs must
; be left alone in region-instrumented functions.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @llvm.va_start.p0(ptr)
declare void @llvm.va_end.p0(ptr)

define i32 @va_noinstr(i32 %n, ...) disable_sanitizer_instrumentation {
entry:
  %ap = alloca [1 x { i32, i32, ptr, ptr }]
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.va_start.p0(ptr %ap)
  %v = va_arg ptr %ap, i32
  call void @llvm.va_end.p0(ptr %ap)
  call void @llvm.sanitizer.instrumentation.end()
  ret i32 %v
}

; The context state must not be read before the first begin() stores it. The
; first thing touching the slot is the store from __msan_get_context_state().
; CHECK-LABEL: @va_noinstr(
; CHECK: %msan_context_state = alloca ptr
; CHECK-NOT: load ptr, ptr %msan_context_state
; CHECK-NOT: va_arg_overflow_size
; CHECK-NOT: va_arg_shadow
; CHECK: call ptr @__msan_get_context_state()
; CHECK: store ptr {{.*}}, ptr %msan_context_state
