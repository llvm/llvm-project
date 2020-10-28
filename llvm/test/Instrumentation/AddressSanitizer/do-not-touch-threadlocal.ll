; RUN: opt < %s -asan -asan-module -enable-new-pm=0 -S | FileCheck %s
; RUN: opt < %s -passes='asan-pipeline' -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
; no action should be taken for thread locals
@xxx = thread_local global i32 0, align 4
; CHECK-NOT: {{call.*__asan_register_globals}}
