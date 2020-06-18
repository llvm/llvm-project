; RUN: llc -verify-machineinstrs < %s | FileCheck %s
; This file contains a collection of basic tests to ensure we didn't
; screw up normal call lowering when a statepoint is a GC transition.

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

declare zeroext i1 @return_i1()
declare zeroext i32 @return_i32()
declare zeroext i32 @return_i32_with_args(i32, i8*)
declare i32* @return_i32ptr()
declare float @return_float()
declare void @varargf(i32, ...)

define i1 @test_i1_return() gc "statepoint-example" {
; CHECK-LABEL: test_i1_return
; This is just checking that a i1 gets lowered normally when there's no extra
; state arguments to the statepoint
; CHECK: pushq %rax
; CHECK: callq return_i1
; CHECK: popq %rcx
; CHECK: retq
entry:
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 1, i32 0, i32 0)
  %call1 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  ret i1 %call1
}

define i32 @test_i32_return() gc "statepoint-example" {
; CHECK-LABEL: test_i32_return
; CHECK: pushq %rax
; CHECK: callq return_i32
; CHECK: popq %rcx
; CHECK: retq
entry:
  %safepoint_token = tail call token (i64, i32, i32 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32f(i64 0, i32 0, i32 ()* @return_i32, i32 0, i32 1, i32 0, i32 0)
  %call1 = call zeroext i32 @llvm.experimental.gc.result.i32(token %safepoint_token)
  ret i32 %call1
}

define i32* @test_i32ptr_return() gc "statepoint-example" {
; CHECK-LABEL: test_i32ptr_return
; CHECK: pushq %rax
; CHECK: callq return_i32ptr
; CHECK: popq %rcx
; CHECK: retq
entry:
  %safepoint_token = tail call token (i64, i32, i32* ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_p0i32f(i64 0, i32 0, i32* ()* @return_i32ptr, i32 0, i32 1, i32 0, i32 0)
  %call1 = call i32* @llvm.experimental.gc.result.p0i32(token %safepoint_token)
  ret i32* %call1
}

define float @test_float_return() gc "statepoint-example" {
; CHECK-LABEL: test_float_return
; CHECK: pushq %rax
; CHECK: callq return_float
; CHECK: popq %rax
; CHECK: retq
entry:
  %safepoint_token = tail call token (i64, i32, float ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_f32f(i64 0, i32 0, float ()* @return_float, i32 0, i32 1, i32 0, i32 0)
  %call1 = call float @llvm.experimental.gc.result.f32(token %safepoint_token)
  ret float %call1
}

define i1 @test_relocate(i32 addrspace(1)* %a) gc "statepoint-example" {
; CHECK-LABEL: test_relocate
; Check that an ununsed relocate has no code-generation impact
; CHECK: pushq %rax
; CHECK: callq return_i1
; CHECK-NEXT: .Ltmp4:
; CHECK-NEXT: popq %rcx
; CHECK-NEXT: .cfi_def_cfa_offset 8
; CHECK-NEXT: retq
entry:
  %safepoint_token = tail call token (i64, i32, i1 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i1f(i64 0, i32 0, i1 ()* @return_i1, i32 0, i32 1, i32 0, i32 0, i32 addrspace(1)* %a)
  %call1 = call i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token %safepoint_token, i32 7, i32 7)
  %call2 = call zeroext i1 @llvm.experimental.gc.result.i1(token %safepoint_token)
  ret i1 %call2
}

define void @test_void_vararg() gc "statepoint-example" {
; CHECK-LABEL: test_void_vararg
; Check a statepoint wrapping a *void* returning vararg function works
; CHECK: callq varargf
entry:
  %safepoint_token = tail call token (i64, i32, void (i32, ...)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_isVoidi32varargf(i64 0, i32 0, void (i32, ...)* @varargf, i32 2, i32 1, i32 42, i32 43, i32 0, i32 0)
  ;; if we try to use the result from a statepoint wrapping a
  ;; non-void-returning varargf, we will experience a crash.
  ret void
}

define i32 @test_transition_args() gc "statepoint-example" {
; CHECK-LABEL: test_transition_args
; CHECK: pushq %rax
; CHECK: callq return_i32
; CHECK: popq %rcx
; CHECK: retq
entry:
  %val = alloca i32
  %safepoint_token = call token (i64, i32, i32 ()*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32f(i64 0, i32 0, i32 ()* @return_i32, i32 0, i32 1, i32 0, i32 0) ["gc-transition" (i32* %val, i64 42)]
  %call1 = call i32 @llvm.experimental.gc.result.i32(token %safepoint_token)
  ret i32 %call1
}

define i32 @test_transition_args_2() gc "statepoint-example" {
; CHECK-LABEL: test_transition_args_2
; CHECK: pushq %rax
; CHECK: callq return_i32
; CHECK: popq %rcx
; CHECK: retq
entry:
  %val = alloca i32
  %arg = alloca i8
  %safepoint_token = call token (i64, i32, i32 (i32, i8*)*, i32, i32, ...) @llvm.experimental.gc.statepoint.p0f_i32i32p0i8f(i64 0, i32 0, i32 (i32, i8*)* @return_i32_with_args, i32 2, i32 1, i32 0, i8* %arg, i32 0, i32 0) ["gc-transition" (i32* %val, i64 42)]
  %call1 = call i32 @llvm.experimental.gc.result.i32(token %safepoint_token)
  ret i32 %call1
}

declare token @llvm.experimental.gc.statepoint.p0f_i1f(i64, i32, i1 ()*, i32, i32, ...)
declare i1 @llvm.experimental.gc.result.i1(token)

declare token @llvm.experimental.gc.statepoint.p0f_i32f(i64, i32, i32 ()*, i32, i32, ...)
declare token @llvm.experimental.gc.statepoint.p0f_i32i32p0i8f(i64, i32, i32 (i32, i8*)*, i32, i32, ...)
declare i32 @llvm.experimental.gc.result.i32(token)

declare token @llvm.experimental.gc.statepoint.p0f_p0i32f(i64, i32, i32* ()*, i32, i32, ...)
declare i32* @llvm.experimental.gc.result.p0i32(token)

declare token @llvm.experimental.gc.statepoint.p0f_f32f(i64, i32, float ()*, i32, i32, ...)
declare float @llvm.experimental.gc.result.f32(token)

declare token @llvm.experimental.gc.statepoint.p0f_isVoidi32varargf(i64, i32, void (i32, ...)*, i32, i32, ...)

declare i32 addrspace(1)* @llvm.experimental.gc.relocate.p1i32(token, i32, i32)
