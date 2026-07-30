; This test checks that we are not instrumenting globals
; that we created ourselves.
; RUN: opt < %s -passes=asan -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @_Z3barv() uwtable sanitize_address {
entry:
  %a = alloca i32, align 4
  call void @_Z3fooPi(ptr %a)
  ret void
}

declare void @_Z3fooPi(ptr)
; We create one global string constant for the stack frame above.
; It should have unnamed_addr and align 1.
; Make sure we don't create any other global constants.
; CHECK: = private unnamed_addr constant{{.*}}align 1
; CHECK-NOT: = private unnamed_addr constant
