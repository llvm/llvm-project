; RUN: llc -verify-machineinstrs -stack-symbol-ordering=0 < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

; This test is checking to make sure that we reuse the same stack slots
; for GC values spilled over two different call sites.  Since the order
; of GC arguments differ, niave lowering code would insert loads and 
; stores to rearrange items on the stack.  We need to make sure (for
; performance) that this doesn't happen.
define i32 @back_to_back_calls(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) #1 gc "statepoint-example" {
; CHECK-LABEL: back_to_back_calls
; The exact stores don't matter, but there need to be three stack slots created
; CHECK-DAG: movq	%rdi, 16(%rsp)
; CHECK-DAG: movq	%rdx, 8(%rsp)
; CHECK-DAG: movq	%rsi, (%rsp)
; There should be no more than three moves
; CHECK-NOT: movq
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c)
  %a1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 12)
  %b1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 13)
  %c1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 14)
; CHECK: callq
; This is the key check.  There should NOT be any memory moves here
; CHECK-NOT: movq
  %safepoint_token2 = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 addrspace(1)* %c1, i32 addrspace(1)* %b1, i32 addrspace(1)* %a1)
  %a2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2, i32 12, i32 14)
  %b2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2, i32 12, i32 13)
  %c2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2, i32 12, i32 12)
; CHECK: callq
  ret i32 1
}

; This test simply checks that minor changes in vm state don't prevent slots
; being reused for gc values.  
define i32 @reserve_first(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) #1 gc "statepoint-example" {
; CHECK-LABEL: reserve_first
; The exact stores don't matter, but there need to be three stack slots created
; CHECK-DAG: movq	%rdi, 16(%rsp)
; CHECK-DAG: movq	%rdx, 8(%rsp)
; CHECK-DAG: movq	%rsi, (%rsp)
  %safepoint_token = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c)
  %a1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 12)
  %b1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 13)
  %c1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 14)
; CHECK: callq
; This is the key check.  There should NOT be any memory moves here
; CHECK-NOT: movq
  %safepoint_token2 = tail call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 addrspace(1)* %a1, i32 0, i32 addrspace(1)* %c1, i32 0, i32 0, i32 addrspace(1)* %c1, i32 addrspace(1)* %b1, i32 addrspace(1)* %a1)
  %a2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2, i32 12, i32 14)
  %b2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2, i32 12, i32 13)
  %c2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2, i32 12, i32 12)
; CHECK: callq
  ret i32 1
}

; Check that we reuse the same stack slot across multiple calls.  The use of
; more than two calls here is critical.  We've had a bug which allowed reuse
; exactly once which went undetected for a long time.
define i32 @back_to_back_deopt(i32 %a, i32 %b, i32 %c) #1 
  gc "statepoint-example" {
; CHECK-LABEL: back_to_back_deopt
; The exact stores don't matter, but there need to be three stack slots created
; CHECK-DAG: movl	%edi, 12(%rsp)
; CHECK-DAG: movl	%esi, 8(%rsp)
; CHECK-DAG: movl	%edx, 4(%rsp)
; CHECK: callq
; CHECK-DAG: movl	%ebx, 12(%rsp)
; CHECK-DAG: movl	%ebp, 8(%rsp)
; CHECK-DAG: movl	%r14d, 4(%rsp)
; CHECK: callq
; CHECK-DAG: movl	%ebx, 12(%rsp)
; CHECK-DAG: movl	%ebp, 8(%rsp)
; CHECK-DAG: movl	%r14d, 4(%rsp)
; CHECK: callq
; CHECK-DAG: movl	%ebx, 12(%rsp)
; CHECK-DAG: movl	%ebp, 8(%rsp)
; CHECK-DAG: movl	%r14d, 4(%rsp)
; CHECK: callq
  call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 3, i32 %a, i32 %b, i32 %c)
call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 3, i32 %a, i32 %b, i32 %c)
call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 3, i32 %a, i32 %b, i32 %c)
call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 3, i32 %a, i32 %b, i32 %c)
  ret i32 1
}

; Test that stack slots are reused for invokes
define i32 @back_to_back_invokes(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c) #1 gc "statepoint-example" personality i32 ()* @"personality_function" {
; CHECK-LABEL: back_to_back_invokes
entry:
  ; The exact stores don't matter, but there need to be three stack slots created
  ; CHECK-DAG: movq	%rdi, 16(%rsp)
  ; CHECK-DAG: movq	%rdx, 8(%rsp)
  ; CHECK-DAG: movq	%rsi, (%rsp)
  ; CHECK: callq
  %safepoint_token = invoke token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i32 addrspace(1)* %a, i32 addrspace(1)* %b, i32 addrspace(1)* %c)
                   to label %normal_return unwind label %exceptional_return

normal_return:
  %a1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 12)
  %b1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 13)
  %c1 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 12, i32 14)
  ; Should work even through bitcasts
  %c1.casted = bitcast i32 addrspace(1)* %c1 to i8 addrspace(1)*
  ; This is the key check.  There should NOT be any memory moves here
  ; CHECK-NOT: movq
  ; CHECK: callq
  %safepoint_token2 = invoke token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* undef, i32 0, i32 0, i32 0, i32 5, i32 0, i32 -1, i32 0, i32 0, i32 0, i8 addrspace(1)* %c1.casted, i32 addrspace(1)* %b1, i32 addrspace(1)* %a1)
                    to label %normal_return2 unwind label %exceptional_return2

normal_return2:
  %a2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2, i32 12, i32 14)
  %b2 = tail call coldcc i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token2, i32 12, i32 13)
  %c2 = tail call coldcc i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token %safepoint_token2, i32 12, i32 12)
  ret i32 1

exceptional_return:
  %landing_pad = landingpad { i8*, i32 }
          cleanup
  ret i32 0

exceptional_return2:
  %landing_pad2 = landingpad { i8*, i32 }
          cleanup
  ret i32 0
}

; Function Attrs: nounwind
declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32) #3
declare i8 addrspace(1)* @llvm.experimental.gc.relocate.p1i8(token, i32, i32) #3

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

declare i32 @"personality_function"()

attributes #1 = { uwtable }
