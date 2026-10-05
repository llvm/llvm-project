; Test that functions not instrumented by AddressSanitizer are still
; instrumented inside llvm.sanitizer.instrumentation.begin()/end() regions, and
; only there.
;
; RUN: opt < %s -passes=asan -asan-kernel=1 \
; RUN:   -asan-instrumentation-with-call-threshold=0 -S | FileCheck %s
; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -S \
; RUN:   | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @callee()

; Only the access between begin() and end() is checked.
define i32 @region(ptr %a, ptr %b, ptr %c) {
entry:
  %x = load i32, ptr %a
  call void @llvm.sanitizer.instrumentation.begin()
  %y = load i32, ptr %b
  call void @llvm.sanitizer.instrumentation.end()
  store i32 %y, ptr %c
  %s = add i32 %x, %y
  ret i32 %s
}

; CHECK-LABEL: define i32 @region(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %x = load i32, ptr %a
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK:         call void @__asan_load4{{(_noabort)?}}(i64 %{{.*}})
; CHECK-NEXT:    %y = load i32, ptr %b
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    store i32 %y, ptr %c
; CHECK-NOT:     __asan_
; CHECK:         ret i32

; The same for functions with disable_sanitizer_instrumentation.
define i32 @region_disable(ptr %a, ptr %b) disable_sanitizer_instrumentation {
entry:
  %x = load i32, ptr %a
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 %x, ptr %b
  call void @llvm.sanitizer.instrumentation.end()
  ret i32 %x
}

; CHECK-LABEL: define i32 @region_disable(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %x = load i32, ptr %a
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK:         call void @__asan_store4{{(_noabort)?}}(i64 %{{.*}})
; CHECK-NEXT:    store i32 %x, ptr %b
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    ret i32 %x

; A block reachable without passing begin() is not in a region.
define i32 @diamond(i1 %cond, ptr %a) {
entry:
  br i1 %cond, label %then, label %join
then:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %join
join:
  %x = load i32, ptr %a
  ret i32 %x
}

; CHECK-LABEL: define i32 @diamond(
; CHECK-NOT:     __asan_
; CHECK:         ret i32

; The inner end() of nested regions does not end the outer region.
define void @nested(ptr %a) {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  store i32 0, ptr %a
  call void @llvm.sanitizer.instrumentation.end()
  store i32 1, ptr %a
  ret void
}

; CHECK-LABEL: define void @nested(
; CHECK:         call void @llvm.sanitizer.instrumentation.end()
; CHECK:         call void @__asan_store4{{(_noabort)?}}(i64 %{{.*}})
; CHECK-NEXT:    store i32 0, ptr %a
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    store i32 1, ptr %a
; CHECK-NEXT:    ret void

; The function's own locals have no redzones and are not checked; neither is
; its stack frame laid out by ASan.
define i32 @locals(ptr %p) {
entry:
  %local = alloca i32
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 1, ptr %local
  %v = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret i32 %v
}

; CHECK-LABEL: define i32 @locals(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %local = alloca i32
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    store i32 1, ptr %local
; CHECK:         call void @__asan_load4{{(_noabort)?}}(i64 %{{.*}})
; CHECK-NEXT:    %v = load i32, ptr %p
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    ret i32 %v

; Locals reached through a phi or select, and memory intrinsics that only touch
; locals, are not checked either; a memcpy between a local and non-local memory
; still is.
define void @locals_indirect(ptr %p, i1 %c) {
entry:
  %a = alloca i32
  %b = alloca [16 x i8]
  call void @llvm.sanitizer.instrumentation.begin()
  br i1 %c, label %then, label %join

then:
  br label %join

join:
  %phi = phi ptr [ %a, %entry ], [ %b, %then ]
  %sel = select i1 %c, ptr %a, ptr %b
  store i32 1, ptr %phi
  store i32 2, ptr %sel
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 4, i1 false)
  call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %p, i64 4, i1 false)
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @locals_indirect(
; CHECK:       join:
; CHECK-NEXT:    %phi = phi ptr
; CHECK-NEXT:    %sel = select i1 %c, ptr %a, ptr %b
; CHECK-NEXT:    store i32 1, ptr %phi
; CHECK-NEXT:    store i32 2, ptr %sel
; CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr %b, ptr %a, i64 4, i1 false)
; CHECK-NEXT:    call ptr @{{(__asan_)?}}memcpy(ptr %b, ptr %p, i64 4)
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()

; A call between two regions may free memory, so the second access to the
; same address is checked again.
define void @call_between_regions(ptr %p) {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 0, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  call void @callee()
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 1, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @call_between_regions(
; CHECK:         call void @__asan_store4{{(_noabort)?}}(i64 %{{.*}})
; CHECK-NEXT:    store i32 0, ptr %p
; CHECK:         call void @callee()
; CHECK:         call void @__asan_store4{{(_noabort)?}}(i64 %{{.*}})
; CHECK-NEXT:    store i32 1, ptr %p

; Memory intrinsics are only replaced inside regions.
define void @memcpy(ptr %d, ptr %s) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 16, i1 false)
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 16, i1 false)
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @memcpy(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %s, i64 16, i1 false)
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    call ptr @{{(__asan_)?}}memcpy(ptr %d, ptr %s, i64 16)
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()

; Without regions, functions lacking sanitize_address are left alone.
define i32 @no_region(ptr %a) {
entry:
  %x = load i32, ptr %a
  ret i32 %x
}

; CHECK-LABEL: define i32 @no_region(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %x = load i32, ptr %a
; CHECK-NEXT:    ret i32 %x

; Fully instrumented functions are unaffected by regions.
define i32 @sanitized(ptr %a, ptr %b) sanitize_address {
entry:
  %x = load i32, ptr %a
  call void @llvm.sanitizer.instrumentation.begin()
  %y = load i32, ptr %b
  call void @llvm.sanitizer.instrumentation.end()
  %s = add i32 %x, %y
  ret i32 %s
}

; CHECK-LABEL: define i32 @sanitized(
; CHECK:         call void @__asan_load4{{(_noabort)?}}(i64 %{{.*}})
; CHECK-NEXT:    %x = load i32, ptr %a
; CHECK:         call void @__asan_load4{{(_noabort)?}}(i64 %{{.*}})
; CHECK-NEXT:    %y = load i32, ptr %b
