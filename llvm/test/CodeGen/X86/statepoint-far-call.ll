; RUN: llc < %s | FileCheck %s
; Test to check that Statepoints with X64 far-immediate targets
; are lowered correctly to an indirect call via a scratch register.

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-win64"

define void @test_far_call() gc "statepoint-example" {
; CHECK-LABEL: test_far_call
; CHECK: pushq %rax
; CHECK: movabsq $140727162896504, %rax 
; CHECK: callq *%rax
; CHECK: popq %rax
; CHECK: retq

entry:
  %safepoint_token = call token (i64, i32, void ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidf(i64 0, i32 0, void ()* inttoptr (i64 140727162896504 to void ()*), i32 0, i32 0, i32 0, i32 0)  
  ret void
}

declare token @llvm.experimental.gc.statepoint.p0f_isVoidf(i64, i32, void ()*, i32, i32, ...)

