; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan | FileCheck %s

; In a disable_sanitizer_instrumentation function, code outside the regions is
; left exactly as it is: no KMSAN runtime calls, no stronger atomic orderings,
; no shadow computations. Inside a region the function is instrumented like a
; sanitize_memory one, including poisoning undef values.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @llvm.masked.store.v4i32.p0(<4 x i32>, ptr, i32, <4 x i1>)

define i32 @outside(ptr %p, <4 x i32> %v, <4 x i1> %m) disable_sanitizer_instrumentation {
entry:
  %a = load atomic i32, ptr %p monotonic, align 4
  store atomic i32 1, ptr %p monotonic, align 4
  %r = atomicrmw add ptr %p, i32 1 monotonic
  %x = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic
  call void @llvm.masked.store.v4i32.p0(<4 x i32> %v, ptr %p, i32 4, <4 x i1> %m)
  %b = xor i32 %a, %r
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 undef, ptr %p, align 4
  call void @llvm.sanitizer.instrumentation.end()
  ret i32 %b
}

; CHECK-LABEL: define i32 @outside(
; CHECK-NEXT: entry:
; CHECK-NEXT: %msan_context_state = alloca ptr
; CHECK-NEXT: store ptr null, ptr %msan_context_state
; CHECK-NEXT: call void @llvm.donothing()
; CHECK-NEXT: %a = load atomic i32, ptr %p monotonic, align 4
; CHECK-NEXT: store atomic i32 1, ptr %p monotonic, align 4
; CHECK-NEXT: %r = atomicrmw add ptr %p, i32 1 monotonic
; CHECK-NEXT: %x = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic
; CHECK-NEXT: call void @llvm.masked.store.v4i32.p0(<4 x i32> %v, ptr align 4 %p, <4 x i1> %m)
; CHECK-NEXT: %b = xor i32 %a, %r
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: call ptr @__msan_get_context_state()
; CHECK-NEXT: store ptr
; CHECK-NEXT: [[SP:%.*]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_4(ptr %p)
; CHECK-NEXT: [[SHP:%.*]] = extractvalue { ptr, ptr } [[SP]], 0
; CHECK-NEXT: extractvalue
; CHECK-NEXT: store i32 -1, ptr [[SHP]], align 4
; CHECK:      store i32 undef, ptr %p, align 4
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT: ret i32 %b
