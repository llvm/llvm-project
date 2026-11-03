; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s

; Basic KMSAN handling of llvm.sanitizer.instrumentation.begin()/end() regions.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare i64 @dummy_call(i64)

define void @instrumented_function() sanitize_memory {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

define void @no_sanitize_function() {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; Outside disable_sanitizer_instrumentation functions the markers mean nothing
; to KMSAN: the context is fetched once in the prologue, not at begin().
; CHECK-LABEL: @instrumented_function(
; CHECK: call ptr @__msan_get_context_state()
; CHECK-NOT: call ptr @__msan_get_context_state()
; CHECK: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT: ret void

; CHECK-LABEL: @no_sanitize_function(
; CHECK: call ptr @__msan_get_context_state()
; CHECK-NOT: call ptr @__msan_get_context_state()
; CHECK: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT: ret void

; Only the accesses inside the region are instrumented. The region does not
; use the context state (loads and stores go through
; __msan_metadata_ptr_for_*), so it is not fetched at all.
define void @noinstr_function(ptr %p, i64 %val) disable_sanitizer_instrumentation {
entry:
  %x = load i64, ptr %p
  %add = add i64 %x, %val
  call void @llvm.sanitizer.instrumentation.begin()
  %y = load i64, ptr %p
  %add2 = add i64 %y, %add
  store i64 %add2, ptr %p
  %z = load i64, ptr %p
  store i64 %z, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  %w = load i64, ptr %p
  ret void
}

; CHECK-LABEL: @noinstr_function(
; CHECK-NOT: msan_context_state
; CHECK: %x = load i64, ptr %p
; CHECK-NEXT: %add = add i64 %x, %val
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: %y = load i64, ptr %p
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: [[YS:%_msld.*]] = load i64, ptr
; %add was computed outside the region, so its shadow is clean.
; CHECK: [[ADD2S:%.*]] = or i64 [[YS]], 0
; CHECK: %add2 = add i64 %y, %add
; CHECK: [[SP1:%.*]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: [[SHP1:%.*]] = extractvalue { ptr, ptr } [[SP1]], 0
; CHECK: store i64 [[ADD2S]], ptr [[SHP1]]
; CHECK: store i64 %add2, ptr %p
; CHECK: %z = load i64, ptr %p
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: [[ZS:%_msld.*]] = load i64, ptr
; CHECK: [[SP2:%.*]] = call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: [[SHP2:%.*]] = extractvalue { ptr, ptr } [[SP2]], 0
; CHECK: store i64 [[ZS]], ptr [[SHP2]]
; CHECK: store i64 %z, ptr %p
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT: %w = load i64, ptr %p
; CHECK-NEXT: ret void
; CHECK-NOT: msan_context_state

define void @noinstr_with_branch(ptr %p, i1 %cond) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  br i1 %cond, label %if.then, label %if.else

if.then:
  %a = load i64, ptr %p
  store i64 %a, ptr %p
  br label %if.end

if.else:
  %b = load i64, ptr %p
  store i64 %b, ptr %p
  br label %if.end

if.end:
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: @noinstr_with_branch(
; CHECK: if.then:
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: if.else:
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)

define void @noinstr_with_loop(ptr %p, i64 %n) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %c = load i64, ptr %p
  store i64 %c, ptr %p
  %inc = add i64 %i, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: @noinstr_with_loop(
; CHECK: loop:
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: exit:

; A call inside the region needs the context state for the parameter and
; return value shadow: it is fetched at begin() and kept in a stack slot.
; Nothing outside the region is touched.
define i64 @noinstr_with_various_insts(ptr %p, i64 %val) disable_sanitizer_instrumentation {
entry:
  %a = alloca i64
  %rmw1 = atomicrmw add ptr %p, i64 %val seq_cst
  %call1 = call i64 @dummy_call(i64 %rmw1)
  store i64 %call1, ptr %a

  call void @llvm.sanitizer.instrumentation.begin()
  %rmw2 = atomicrmw add ptr %p, i64 %val seq_cst
  %call2 = call i64 @dummy_call(i64 %rmw2)
  store i64 %call2, ptr %a
  call void @llvm.sanitizer.instrumentation.end()

  %rmw3 = atomicrmw add ptr %p, i64 %val seq_cst
  %call3 = call i64 @dummy_call(i64 %rmw3)
  store i64 %call3, ptr %a
  ret i64 %call3
}

; CHECK-LABEL: @noinstr_with_various_insts(
; CHECK: %msan_context_state = alloca ptr
; CHECK-NEXT: store ptr null, ptr %msan_context_state
; CHECK-NOT: __msan_metadata
; CHECK-NOT: __msan_get_context_state
; CHECK: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: [[CTX:%.*]] = call ptr @__msan_get_context_state()
; CHECK-NEXT: store ptr [[CTX]], ptr %msan_context_state
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: load ptr, ptr %msan_context_state
; CHECK: %param_shadow = getelementptr
; CHECK: %call2 = call i64 @dummy_call(i64 %rmw2)
; CHECK: call void @llvm.sanitizer.instrumentation.end()
; CHECK-NOT: __msan_metadata
; CHECK-NOT: __msan_get_context_state
; CHECK: ret i64 %call3

; A nested begin() is dominated by the outer one, which has already fetched
; the context; the context is fetched once.
define void @nested_call(i64 %val) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.begin()
  %call = call i64 @dummy_call(i64 %val)
  call void @llvm.sanitizer.instrumentation.end()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: @nested_call(
; CHECK: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT: call ptr @__msan_get_context_state()
; CHECK-NEXT: store ptr {{.*}}, ptr %msan_context_state
; CHECK-NEXT: call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NOT: __msan_get_context_state
; CHECK: %call = call i64 @dummy_call(i64 %val)

; A begin() at the start of a block that is entered from inside a region on
; one path and from outside on the other is not nested: the block starts
; outside the region (must-join), so this begin() has to fetch the context.
; (The input is unbalanced and diagnosed, but must not miscompile.)
define void @begin_after_partial_join(i64 %val, i1 %c) disable_sanitizer_instrumentation {
entry:
  br i1 %c, label %a, label %b

a:
  br label %join

b:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %join

join:
  call void @llvm.sanitizer.instrumentation.begin()
  %call = call i64 @dummy_call(i64 %val)
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: @begin_after_partial_join(
; CHECK:       join:
; CHECK-NEXT:    call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    call ptr @__msan_get_context_state()
; CHECK-NEXT:    store ptr {{.*}}, ptr %msan_context_state
; CHECK:         %call = call i64 @dummy_call(i64 %val)
