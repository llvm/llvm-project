; RUN: opt < %s -asan -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
@xxx = global i32 0, align 4

; If a global is present, __asan_[un]register_globals should be called from
; module ctor/dtor

; CHECK: llvm.global_ctors
; CHECK: llvm.global_dtors

; CHECK: define internal void @asan.module_ctor
; CHECK-NOT: ret
; CHECK: call void @__asan_register_globals
; CHECK: ret

; CHECK: define internal void @asan.module_dtor
; CHECK-NOT: ret
; CHECK: call void @__asan_unregister_globals
; CHECK: ret
