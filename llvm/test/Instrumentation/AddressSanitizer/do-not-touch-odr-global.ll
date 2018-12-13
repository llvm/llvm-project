; This test checks that we instrument regular globals, but do not touch
; the linkonce_odr ones.
; RUN: opt < %s -asan -asan-module -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"
; no action should be taken for these globals
@global_noinst = linkonce_odr constant [2 x i8] [i8 1, i8 2]
@global_weak_noinst = weak_odr constant [2 x i8] [i8 1, i8 2]
@global_inst = private constant [2 x i8] [i8 1, i8 2]
; CHECK-NOT: {{asan_gen.*global_noinst}}
; CHECK-NOT: {{asan_gen.*global_weak_noinst}}
; CHECK: {{asan_gen.*global_inst}}
; CHECK: @asan.module_ctor
