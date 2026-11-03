; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s

; Varargs in a region-instrumented function.
;
; The callee side (va_start/va_copy) is not instrumented: the VarArgHelpers copy
; va_arg_tls in the function prologue, which is outside every region, so
; %msan_context_state has not been stored to yet and the emitted code would
; load that slot and then read and write through the resulting pointer.
;
; The caller side is instrumented: a call to a variadic function inside a
; region publishes the shadow of the variadic arguments and their total size,
; like any other in-region call publishes param_shadow. Otherwise an
; instrumented variadic callee would consume whatever the last variadic call
; left in the context.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @llvm.va_start.p0(ptr)
declare void @llvm.va_end.p0(ptr)
declare i32 @vprintf_like(ptr, ...)

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

; Nothing in this function uses the context state, so it is not even fetched;
; in particular no va_arg_tls is copied in the prologue.
; CHECK-LABEL: @va_noinstr(
; CHECK-NOT: msan_context_state
; CHECK-NOT: va_arg_overflow_size
; CHECK-NOT: va_arg_shadow
; CHECK: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NOT: msan_context_state
; CHECK-NOT: va_arg_overflow_size
; CHECK-NOT: va_arg_shadow
; CHECK: ret i32

define void @va_call(ptr %fmt, ptr %p) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  %x = load i64, ptr %p
  call i32 (ptr, ...) @vprintf_like(ptr %fmt, i64 %x)
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: @va_call(
; CHECK: call ptr @__msan_get_context_state()
; CHECK: [[XS:%_msld.*]] = load i64, ptr
; CHECK: %va_arg_shadow = getelementptr {{.*}}, ptr {{%.*}}, i32 0, i32 2
; CHECK: store i64 [[XS]], ptr {{%.*}}
; CHECK: %va_arg_overflow_size = getelementptr {{.*}}, ptr {{%.*}}, i32 0, i32 4
; CHECK: store i64 0, ptr %va_arg_overflow_size
; CHECK: call i32 (ptr, ...) @vprintf_like(ptr %fmt, i64 %x)
