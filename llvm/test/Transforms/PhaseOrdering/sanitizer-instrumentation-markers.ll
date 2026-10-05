; RUN: opt -passes='default<O2>' -S < %s | FileCheck %s

; The sanitizer instrumentation region markers may access any memory that is
; visible outside the function, so the optimizer must not move accesses to such
; memory into or out of a region, and must not merge, hoist, sink or delete the
; markers themselves.

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @callee(i32)

define i32 @no_cse_across_markers(ptr %p) disable_sanitizer_instrumentation {
; CHECK-LABEL: define i32 @no_cse_across_markers(
; CHECK:         %a = load i32, ptr %p
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    %b = load i32, ptr %p
; CHECK-NEXT:    store i32 1, ptr %p
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    store i32 2, ptr %p
  %a = load i32, ptr %p
  call void @llvm.sanitizer.instrumentation.begin()
  %b = load i32, ptr %p
  store i32 1, ptr %p
  call void @llvm.sanitizer.instrumentation.end()
  store i32 2, ptr %p
  %r = add i32 %a, %b
  ret i32 %r
}

define void @no_licm_out_of_region(ptr %p, i32 %n) disable_sanitizer_instrumentation {
; CHECK-LABEL: define void @no_licm_out_of_region(
; CHECK:       loop:
; CHECK:         tail call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    %v = load i32, ptr %p
; CHECK-NEXT:    tail call void @callee(i32 %v)
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.end()
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @llvm.sanitizer.instrumentation.begin()
  %v = load i32, ptr %p
  call void @callee(i32 %v)
  call void @llvm.sanitizer.instrumentation.end()
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

define void @no_merge(i1 %c, i32 %x) disable_sanitizer_instrumentation {
; CHECK-LABEL: define void @no_merge(
; CHECK:       a:
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    tail call void @callee(i32 %x)
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.end()
; CHECK:       b:
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    tail call void @callee(i32 %x)
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.end()
entry:
  br i1 %c, label %a, label %b
a:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @callee(i32 %x)
  call void @llvm.sanitizer.instrumentation.end()
  br label %exit
b:
  call void @llvm.sanitizer.instrumentation.begin()
  call void @callee(i32 %x)
  call void @llvm.sanitizer.instrumentation.end()
  br label %exit
exit:
  ret void
}

define void @empty_region() disable_sanitizer_instrumentation {
; CHECK-LABEL: define void @empty_region(
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    ret void
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}
