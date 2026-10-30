; RUN: opt -passes='default<O2>' -S < %s | FileCheck %s

; The sanitizer instrumentation region markers have the semantics of an
; `asm volatile` with no operands or clobbers: the optimizer must not delete
; them or move calls that may access memory across them. They are not memory
; barriers, so ordinary loads and stores are not checked here.

declare void @llvm.sanitizer.instrumentation.begin()
declare void @llvm.sanitizer.instrumentation.end()
declare void @callee(i32)

; A call inside a region stays inside it, also in a loop where the region could
; otherwise be hoisted or sunk around it.
define void @call_stays_in_region(i32 %x, i32 %n) disable_sanitizer_instrumentation {
; CHECK-LABEL: define void @call_stays_in_region(
; CHECK:       loop:
; CHECK:         tail call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    tail call void @callee(i32 %x)
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.end()
entry:
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %i.next, %loop ]
  call void @llvm.sanitizer.instrumentation.begin()
  call void @callee(i32 %x)
  call void @llvm.sanitizer.instrumentation.end()
  %i.next = add i32 %i, 1
  %c = icmp slt i32 %i.next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}

; Empty regions are not deleted.
define void @empty_region() disable_sanitizer_instrumentation {
; CHECK-LABEL: define void @empty_region(
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.begin()
; CHECK-NEXT:    tail call void @llvm.sanitizer.instrumentation.end()
; CHECK-NEXT:    ret void
  call void @llvm.sanitizer.instrumentation.begin()
  call void @llvm.sanitizer.instrumentation.end()
  ret void
}
