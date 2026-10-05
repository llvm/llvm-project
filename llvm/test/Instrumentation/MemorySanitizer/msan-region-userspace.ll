; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s

; Regions work the same way under userspace MemorySanitizer, which addresses
; its TLS directly and so needs no context state: nothing is emitted outside
; the region, and inside it accesses are instrumented and calls publish their
; argument shadow.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @use(i32)

define i32 @region_in_userspace_msan(ptr %p, i32 %x) disable_sanitizer_instrumentation {
entry:
  %a = alloca i32
  store i32 %x, ptr %a
  %w = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.begin()
  %v = load i32, ptr %p
  %l = load i32, ptr %a
  call void @use(i32 %l)
  call void @llvm.sanitizer.instrumentation.end()
  %z = load i32, ptr %p
  ret i32 %v
}

; CHECK-LABEL: @region_in_userspace_msan(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.donothing()
; CHECK-NEXT:    %a = alloca i32
; CHECK-NEXT:    store i32 %x, ptr %a
; CHECK-NEXT:    %w = load i32, ptr %p
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; The out-of-region local the region reads is unpoisoned on entry.
; CHECK-NEXT:    ptrtoint ptr %a to i64
; CHECK-NEXT:    xor
; CHECK-NEXT:    inttoptr
; CHECK-NEXT:    call void @llvm.memset.p0.i64(ptr align 4 {{%.*}}, i8 0, i64 4, i1 false)
; CHECK-NEXT:    %v = load i32, ptr %p
; CHECK-NEXT:    ptrtoint ptr %p to i64
; CHECK-NEXT:    xor
; CHECK-NEXT:    inttoptr
; CHECK-NEXT:    load i32, ptr
; CHECK:         %l = load i32, ptr %a
; CHECK:         [[LS:%.*]] = load i32, ptr
; CHECK-NEXT:    store i32 [[LS]], ptr @__msan_param_tls
; CHECK-NEXT:    call void @use(i32 %l)
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; The return is outside the region: no retval shadow is published.
; CHECK-NEXT:    %z = load i32, ptr %p
; CHECK-NEXT:    ret i32 %v
; CHECK-NOT:     __msan_get_context_state
; CHECK-NOT:     msan_context_state
