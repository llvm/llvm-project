; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @instrumented_function() sanitize_memory {
entry:
  call void @llvm.kmsan.instrumentation.begin()
  call void @llvm.kmsan.instrumentation.end()
  ret void
}
; CHECK: warning: {{.*}}llvm.kmsan.instrumentation intrinsics are ignored in fully instrumented or no_sanitize("memory") functions

define void @no_sanitize_function() {
entry:
  call void @llvm.kmsan.instrumentation.begin()
  call void @llvm.kmsan.instrumentation.end()
  ret void
}
; CHECK: warning: {{.*}}llvm.kmsan.instrumentation intrinsics are ignored in fully instrumented or no_sanitize("memory") functions

define void @noinstr_function(ptr %p, i64 %val) disable_sanitizer_instrumentation {
entry:
  %x = load i64, ptr %p
  %add = add i64 %x, %val
  call void @llvm.kmsan.instrumentation.begin()
  %y = load i64, ptr %p
  %add2 = add i64 %y, %add
  store i64 %add2, ptr %p
  call void @llvm.kmsan.instrumentation.update.context()
  %z = load i64, ptr %p
  store i64 %z, ptr %p
  call void @llvm.kmsan.instrumentation.end()
  %w = load i64, ptr %p
  ret void
}

; CHECK-LABEL: @noinstr_function(
; CHECK-NOT: __msan_get_context_state
; CHECK: %x = load i64, ptr %p
; CHECK: %add = add i64 %x, %val
; CHECK: [[CTX1:%.*]] = call ptr @__msan_get_context_state()
; CHECK: store ptr [[CTX1]], ptr %msan_context_state
; CHECK: %y = load i64, ptr %p
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: %add2 = add i64 %y, %add
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: store i64 %add2, ptr %p
; CHECK: [[CTX2:%.*]] = call ptr @__msan_get_context_state()
; CHECK: store ptr [[CTX2]], ptr %msan_context_state
; CHECK: %z = load i64, ptr %p
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: store i64 %z, ptr %p
; CHECK-NOT: call ptr @__msan_get_context_state()
; CHECK-NOT: call { ptr, ptr } @__msan_metadata_ptr_for_load_8
; CHECK: %w = load i64, ptr %p
; CHECK: ret void

declare void @llvm.kmsan.instrumentation.begin()
declare void @llvm.kmsan.instrumentation.end()
declare void @llvm.kmsan.instrumentation.update.context()

define void @noinstr_with_branch(ptr %p, i1 %cond) disable_sanitizer_instrumentation {
entry:
  call void @llvm.kmsan.instrumentation.begin()
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
  call void @llvm.kmsan.instrumentation.end()
  ret void
}

; CHECK-LABEL: @noinstr_with_branch(
; CHECK: [[CTX3:%.*]] = call ptr @__msan_get_context_state()
; CHECK: store ptr [[CTX3]], ptr %msan_context_state
; CHECK: if.then:
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: if.else:
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)

define void @noinstr_with_loop(ptr %p, i64 %n) disable_sanitizer_instrumentation {
entry:
  call void @llvm.kmsan.instrumentation.begin()
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %c = load i64, ptr %p
  store i64 %c, ptr %p
  %inc = add i64 %i, 1
  %cmp = icmp ult i64 %inc, %n
  br i1 %cmp, label %loop, label %exit

exit:
  call void @llvm.kmsan.instrumentation.end()
  ret void
}

; CHECK-LABEL: @noinstr_with_loop(
; CHECK: [[CTX4:%.*]] = call ptr @__msan_get_context_state()
; CHECK: store ptr [[CTX4]], ptr %msan_context_state
; CHECK: loop:
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8(ptr %p)
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: exit:

declare i64 @dummy_call(i64)

define i64 @noinstr_with_various_insts(ptr %p, i64 %val) disable_sanitizer_instrumentation {
entry:
  %a = alloca i64
  %rmw1 = atomicrmw add ptr %p, i64 %val seq_cst
  %call1 = call i64 @dummy_call(i64 %rmw1)
  store i64 %call1, ptr %a
  
  call void @llvm.kmsan.instrumentation.begin()
  %rmw2 = atomicrmw add ptr %p, i64 %val seq_cst
  %call2 = call i64 @dummy_call(i64 %rmw2)
  store i64 %call2, ptr %a
  call void @llvm.kmsan.instrumentation.end()
  
  %rmw3 = atomicrmw add ptr %p, i64 %val seq_cst
  %call3 = call i64 @dummy_call(i64 %rmw3)
  store i64 %call3, ptr %a
  ret i64 %call3
}

; CHECK-LABEL: @noinstr_with_various_insts(
; CHECK-NOT: __msan_metadata
; CHECK: call ptr @__msan_get_context_state()
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_store_8(ptr %p)
; CHECK: ret i64 %call3