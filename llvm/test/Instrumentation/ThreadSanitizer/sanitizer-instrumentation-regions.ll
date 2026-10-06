; Test that functions that are not instrumented as a whole are instrumented
; inside llvm.sanitizer.instrumentation.begin()/end() regions.
; RUN: opt < %s -passes='function(tsan),module(tsan-module)' -S | FileCheck %s
; KCSAN disables the entry/exit hooks.
; RUN: opt < %s -passes='function(tsan),module(tsan-module)' \
; RUN:   -tsan-instrument-func-entry-exit=0 -tsan-distinguish-volatile=1 -S \
; RUN:   | FileCheck %s --check-prefix=KCSAN

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)
declare void @foo()

; In a disable_sanitizer_instrumentation function, plain and atomic accesses
; and memory intrinsics are instrumented only inside regions, and there are no
; entry/exit hooks.
define i32 @region(ptr %a, ptr %b, ptr %d) disable_sanitizer_instrumentation {
entry:
  %out.load = load i32, ptr %a
  %out.atomic = load atomic i32, ptr %b seq_cst, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %a, i64 16, i1 false)
  call void @llvm.sanitizer.instrumentation.begin()
  %in.load = load i32, ptr %a
  store i32 %in.load, ptr %b
  %in.atomic = load atomic i32, ptr %b seq_cst, align 4
  store volatile i32 %in.atomic, ptr %d
  call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %a, i64 16, i1 false)
  call void @llvm.sanitizer.instrumentation.end()
  store i32 %out.load, ptr %b
  store atomic i32 %out.atomic, ptr %b seq_cst, align 4
  ret i32 %in.load
}

; CHECK-LABEL: define i32 @region(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %out.load = load i32, ptr %a
; CHECK-NEXT:    %out.atomic = load atomic i32, ptr %b seq_cst
; CHECK-NEXT:    call void @llvm.memcpy.p0.p0.i64(ptr %d, ptr %a, i64 16, i1 false)
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    call void @__tsan_read4(ptr %a)
; CHECK-NEXT:    %in.load = load i32, ptr %a
; CHECK-NEXT:    call void @__tsan_write4(ptr %b)
; CHECK-NEXT:    store i32 %in.load, ptr %b
; CHECK-NEXT:    %0 = call i32 @__tsan_atomic32_load(ptr %b, i32 5)
; CHECK:         call void @__tsan_write4(ptr %d)
; CHECK-NEXT:    store volatile i32
; CHECK-NEXT:    call ptr @__tsan_memcpy(ptr %d, ptr %a, i64 16)
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    store i32 %out.load, ptr %b
; CHECK-NEXT:    store atomic i32 %out.atomic, ptr %b seq_cst
; CHECK-NEXT:    ret i32 %in.load

; KCSAN-LABEL: define i32 @region(
; KCSAN:         call void @llvm.sanitizer.instrumentation.begin()
; KCSAN-NEXT:    call void @__tsan_read4(ptr %a)
; KCSAN:         call void @__tsan_volatile_write4(ptr %d)
; KCSAN:         call void @llvm.sanitizer.instrumentation.end()
; KCSAN-NOT:     call {{.*}}@__tsan_
; KCSAN:         ret i32 %in.load

; A block reachable without passing through begin() is not in a region.
define void @diamond(i1 %c, ptr %a) disable_sanitizer_instrumentation {
entry:
  br i1 %c, label %enter, label %join

enter:
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 1, ptr %a
  br label %join

join:
  store i32 2, ptr %a
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @diamond(
; CHECK:       enter:
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    call void @__tsan_write4(ptr %a)
; CHECK-NEXT:    store i32 1, ptr %a
; CHECK:       join:
; CHECK-NEXT:    store i32 2, ptr %a
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    ret void

; Regions nest.
define void @nested(ptr %a) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  store i32 1, ptr %a
  call void @llvm.sanitizer.instrumentation.end()
  store i32 2, ptr %a
  ret void
}

; CHECK-LABEL: define void @nested(
; CHECK:         call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    call void @__tsan_write4(ptr %a)
; CHECK-NEXT:    store i32 1, ptr %a
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    store i32 2, ptr %a
; CHECK-NEXT:    ret void

; A call outside the regions separates the accesses on either side of it, so
; the read before the write is not folded into the write.
define void @call_between_regions(ptr %a) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  %v = load i32, ptr %a
  call void @llvm.sanitizer.instrumentation.end()
  call void @foo()
  call void @llvm.sanitizer.instrumentation.begin()
  store i32 %v, ptr %a
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: define void @call_between_regions(
; CHECK:         call void @__tsan_read4(ptr %a)
; CHECK:         call void @foo()
; CHECK:         call void @__tsan_write4(ptr %a)

; Functions without sanitize_thread still get their atomics and entry/exit
; hooks instrumented everywhere; regions add plain accesses.
define i32 @no_sanitize_thread(ptr %a) {
entry:
  %out.load = load i32, ptr %a
  call void @llvm.sanitizer.instrumentation.begin()
  %in.load = load i32, ptr %a
  call void @llvm.sanitizer.instrumentation.end()
  %atomic = load atomic i32, ptr %a seq_cst, align 4
  %r = add i32 %out.load, %in.load
  ret i32 %r
}

; CHECK-LABEL: define i32 @no_sanitize_thread(
; CHECK:         call void @__tsan_func_entry(
; CHECK-NEXT:    %out.load = load i32, ptr %a
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    call void @__tsan_read4(ptr %a)
; CHECK-NEXT:    %in.load = load i32, ptr %a
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    call i32 @__tsan_atomic32_load(ptr %a, i32 5)
; CHECK:         call void @__tsan_func_exit()

; Without regions, disable_sanitizer_instrumentation functions are left alone.
define i32 @no_region(ptr %a) disable_sanitizer_instrumentation {
entry:
  %v = load i32, ptr %a
  %atomic = load atomic i32, ptr %a seq_cst, align 4
  ret i32 %v
}

; CHECK-LABEL: define i32 @no_region(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %v = load i32, ptr %a
; CHECK-NEXT:    %atomic = load atomic i32, ptr %a seq_cst
; CHECK-NEXT:    ret i32 %v

; Instrumented functions are instrumented as usual, regions or not.
define i32 @sanitized(ptr %a) sanitize_thread {
entry:
  %out.load = load i32, ptr %a
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  ret i32 %out.load
}

; CHECK-LABEL: define i32 @sanitized(
; CHECK:         call void @__tsan_func_entry(
; CHECK-NEXT:    call void @__tsan_read4(ptr %a)
; CHECK-NEXT:    %out.load = load i32, ptr %a
