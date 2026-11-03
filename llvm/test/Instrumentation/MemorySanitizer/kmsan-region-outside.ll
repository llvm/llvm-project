; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s

; In a disable_sanitizer_instrumentation function, code outside the regions is
; left exactly as it is: no KMSAN runtime calls, no stronger atomic orderings,
; no shadow computations. Inside a region the function is instrumented like a
; sanitize_memory one, including poisoning undef and poison values.

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
  store i32 poison, ptr %p, align 4
  call void @llvm.sanitizer.instrumentation.end()
  ret i32 %b
}

; CHECK-LABEL: define i32 @outside(
; CHECK-NEXT: entry:
; CHECK-NEXT: call void @llvm.donothing()
; CHECK-NEXT: %a = load atomic i32, ptr %p monotonic, align 4
; CHECK-NEXT: store atomic i32 1, ptr %p monotonic, align 4
; CHECK-NEXT: %r = atomicrmw add ptr %p, i32 1 monotonic
; CHECK-NEXT: %x = cmpxchg ptr %p, i32 0, i32 1 monotonic monotonic
; CHECK-NEXT: call void @llvm.masked.store.v4i32.p0(<4 x i32> %v, ptr align 4 %p, <4 x i1> %m)
; CHECK-NEXT: %b = xor i32 %a, %r
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: [[SP:%.*]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_4(ptr %p)
; CHECK-NEXT: [[SHP:%.*]] = extractvalue { ptr, ptr } [[SP]], 0
; CHECK-NEXT: extractvalue
; CHECK-NEXT: store i32 -1, ptr [[SHP]], align 4
; CHECK:      store i32 poison, ptr %p, align 4
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT: ret i32 %b

; Locals allocated outside the regions get no __msan_poison_alloca, and stores
; to them outside the regions do not update their shadow, so their shadow is
; whatever was left in the stack slot. Those that the regions can see (as the
; underlying object of an in-region pointer operand, e.g. a load or a call
; argument) are unpoisoned at every begin() they dominate, so that reading
; them inside a region does not report; %unused is never referenced inside a
; region and is left alone. Allocas inside a region are poisoned at the alloca
; as usual; an alloca that follows the begin() in the entry block is not
; unpoisoned at that begin().

define void @outside_allocas(i64 %v, i1 %c) disable_sanitizer_instrumentation {
entry:
  %a = alloca i64
  %arr = alloca [4 x i32]
  %unused = alloca i64
  store i64 %v, ptr %a
  store i64 %v, ptr %unused
  call void @llvm.sanitizer.instrumentation.begin()
  %x = load i64, ptr %a
  call void @use(i64 %x)
  %elt = getelementptr [4 x i32], ptr %arr, i64 0, i64 2
  call void @use_ptr(ptr %elt)
  %in = alloca i64
  call void @llvm.sanitizer.instrumentation.end()
  %late = alloca i64
  br i1 %c, label %again, label %exit

again:
  call void @llvm.sanitizer.instrumentation.begin()
  %y = load i64, ptr %late
  call void @use(i64 %y)
  call void @llvm.sanitizer.instrumentation.end()
  br label %exit

exit:
  ret void
}

declare void @use(i64)
declare void @use_ptr(ptr)

; CHECK-LABEL: define void @outside_allocas(
; CHECK:      %a = alloca i64
; CHECK-NEXT: %arr = alloca [4 x i32]
; CHECK-NEXT: %unused = alloca i64
; CHECK-NEXT: store i64 %v, ptr %a
; CHECK-NEXT: store i64 %v, ptr %unused
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: call ptr @__msan_get_context_state()
; CHECK-NEXT: store ptr
; CHECK-NEXT: call void @__msan_unpoison_alloca(ptr %a, i64 8)
; CHECK-NEXT: call void @__msan_unpoison_alloca(ptr %arr, i64 16)
; CHECK-NOT:  __msan_unpoison_alloca
; CHECK:      %in = alloca i64
; CHECK-NEXT: call void @__msan_poison_alloca(ptr %in, i64 8,
; CHECK:      call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT: %late = alloca i64
; CHECK-NEXT: br i1 %c
; CHECK:      again:
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: call ptr @__msan_get_context_state()
; CHECK-NEXT: store ptr
; CHECK-NEXT: call void @__msan_unpoison_alloca(ptr %a, i64 8)
; CHECK-NEXT: call void @__msan_unpoison_alloca(ptr %arr, i64 16)
; CHECK-NEXT: call void @__msan_unpoison_alloca(ptr %late, i64 8)
; CHECK-NEXT: %y = load i64, ptr %late
; CHECK-NEXT: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %late)

; A local whose address escapes outside the regions (here: stored to memory)
; may be reached inside them through that memory, so it is unpoisoned too.
define void @escaped_alloca(ptr %slot) disable_sanitizer_instrumentation {
entry:
  %esc = alloca i64
  %local = alloca i64
  store ptr %esc, ptr %slot
  store i64 1, ptr %local
  call void @llvm.sanitizer.instrumentation.begin()
  %p = load ptr, ptr %slot
  %v = load i64, ptr %p
  call void @use(i64 %v)
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @escaped_alloca(
; CHECK:      call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: call ptr @__msan_get_context_state()
; CHECK-NEXT: store ptr
; CHECK-NEXT: call void @__msan_unpoison_alloca(ptr %esc, i64 8)
; CHECK-NEXT: %p = load ptr, ptr %slot
