; RUN: opt < %s -msan-check-access-address=0 -msan-kernel=1 -S -passes=msan 2>&1 | FileCheck %s

; This test verifies that the MemorySanitizer region instrumentation does not
; generate broken IR (violating dominance constraints) when `update.context` 
; is called after `begin` in complex CFGs.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare void @llvm.kmsan.instrumentation.begin()
declare void @llvm.kmsan.instrumentation.update.context()
declare void @llvm.kmsan.instrumentation.end()

define i1 @do_syscall_64(ptr %regs, i32 %nr) disable_sanitizer_instrumentation {
entry:
  call void @llvm.kmsan.instrumentation.begin()
  br label %loop

loop:
  %x = load i64, ptr %regs
  call void @llvm.kmsan.instrumentation.update.context()
  %y = load i64, ptr %regs
  br i1 false, label %loop, label %exit

exit:
  call void @llvm.kmsan.instrumentation.end()
  ret i1 true
}

; CHECK-LABEL: @do_syscall_64(
; CHECK: entry:
; CHECK: [[CTX_ALLOCA:%.*]] = alloca ptr
; CHECK: [[CTX1:%.*]] = call ptr @__msan_get_context_state()
; CHECK: store ptr [[CTX1]], ptr [[CTX_ALLOCA]]
; CHECK: loop:
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8
; CHECK: [[CTX2:%.*]] = call ptr @__msan_get_context_state()
; CHECK: store ptr [[CTX2]], ptr [[CTX_ALLOCA]]
; CHECK: call { ptr, ptr } @__msan_metadata_ptr_for_load_8