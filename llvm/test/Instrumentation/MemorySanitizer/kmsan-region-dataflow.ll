; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s

; Region membership is a "must" dataflow problem over the region nesting depth.
; These tests cover the cases where an "any path" or flag-based analysis gets it
; wrong.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()

; A block reachable both from inside and from outside a region must not be
; instrumented: on the path that never executed begin(), %msan_context_state
; holds no context state, and the instrumentation would load a wild pointer out
; of it and then store through it.

define i32 @unbalanced_join(ptr %p, i1 %cond) disable_sanitizer_instrumentation {
entry:
  br i1 %cond, label %left, label %right

left:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %merge

right:
  br label %merge

merge:
  %v = load i32, ptr %p
  ret i32 %v
}

; CHECK-LABEL: @unbalanced_join(
; CHECK: merge:
; CHECK-NOT: load ptr, ptr %msan_context_state
; CHECK-NOT: __msan_metadata_ptr_for_load_4
; CHECK: ret i32 %v

; The same join is instrumented once every incoming path has entered a region.

define i32 @balanced_join(ptr %p, i1 %cond) disable_sanitizer_instrumentation {
entry:
  br i1 %cond, label %left, label %right

left:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %merge

right:
  call void @llvm.sanitizer.instrumentation.begin()
  br label %merge

merge:
  %v = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret i32 %v
}

; CHECK-LABEL: @balanced_join(
; CHECK: merge:
; CHECK: %v = load i32, ptr %p
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_4(ptr %p)

; Regions nest: the inner end() must not terminate the outer region.

define void @nested_regions(ptr %p) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.begin()
  %a = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  %b = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  %c = load i32, ptr %p
  ret void
}

; CHECK-LABEL: @nested_regions(
; CHECK: %a = load i32, ptr %p
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_4(ptr %p)
; CHECK: %b = load i32, ptr %p
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_4(ptr %p)
; CHECK: %c = load i32, ptr %p
; CHECK-NOT: __msan_metadata_ptr_for_load_4
; CHECK: ret void

; Nesting also has to survive a join: both predecessors of %merge are at depth
; one, so %merge is still inside the outer region.

define void @nested_regions_join(ptr %p, i1 %cond) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.begin()
  br i1 %cond, label %inner, label %merge

inner:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  br label %merge

merge:
  %v = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}

; CHECK-LABEL: @nested_regions_join(
; CHECK: merge:
; CHECK: %v = load i32, ptr %p
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_4(ptr %p)

; An end() without a matching begin() is malformed input. Saturate the depth at
; zero rather than wrapping around and instrumenting the rest of the function.

define void @unmatched_end(ptr %p) disable_sanitizer_instrumentation {
entry:
  call void @llvm.sanitizer.instrumentation.end()
  %v = load i32, ptr %p
  ret void
}

; CHECK-LABEL: @unmatched_end(
; CHECK-NOT: __msan_metadata_ptr_for_load_4
; CHECK: ret void
